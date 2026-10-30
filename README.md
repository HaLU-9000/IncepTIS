# IncepTIS
This is the official repository for **IncepTIS** and **IncepTIS-evolve**. 
**IncepTIS** is an NTv3 + Attention U-Net model that predicts position-resolved translation initiation signals under normal and stress conditions and is trained on the TISCA2 dataset, which maps translation initiation sites (TISs) under physiological conditions.
**IncepTIS-Evolve** is a tool that uses IncepTIS predictions to optimize sequences for condition-specific translation initiation at desired positions.

> ***Note:***  *As our work is currently available only as a preprint, we haven't yet released model weights or training dataset. Instead, we provide a quick demonstration to illustrate how our model works. These resources will be made available upon official publication, so please stay tuned!* 

### System Requirements
---
IncepTIS is primarily developed and tested on Linux systems.
- **Operating system**: Ubuntu 24.04.3 LTS x86_64 with kernel 6.14.0-37-generic.
- **Hardware**: NVIDIA GPU required for training and inference. The published model was trained on 4 x NVIDIA H100 SXM5 94 GB. GENCODE predictions ran on the same GPUs. All README commands for reproducing the result were verified on one NVIDIA GeForce RTX 5090 32 GB.
- **Python**: Python 3.12.3
- **PyTorch**: PyTorch 2.8.0+cu128
- **CUDA**: PyTorch CUDA 12.8 build
- **NVIDIA driver**: version 580.65.06 (local RTX 5090 host) and 580.105.08 (cluster H100 nodes, CUDA 13.1 capable)
- **cuDNN**: cuDNN bundled in PyTorch 2.8.0+cu128

## Installation
### From Source
---
```bash
git clone https://github.com/HaLU-9000/IncepTIS.git
cd IncepTIS
uv sync --locked
```

### As a Python Package
---
Prediction and sequence design are also available through the Python package.

```bash
uv build --wheel                                              # dist/inceptis-0.4.0-py3-none-any.whl
uv venv --python 3.12 ~/inceptis-env
uv pip install --python ~/inceptis-env/bin/python dist/inceptis-0.4.0-py3-none-any.whl
```
or
```bash
pip install "git+https://github.com/HaLU-9000/IncepTIS.git"
```

The package `inceptis` (version 0.4.0) holds the inference package `inceptis_predict` and the `inceptis-evolve` command (package `inceptis_evolve`); its dependencies, including `torch==2.8.0` with CUDA 12.8, resolve from PyPI. Training, the GENCODE scripts, the design configurations and the demo data are run from the source tree. The installed command and the API load the NTv3 configuration and tokenizer from the Hugging Face cache above, so set the same `HF_HOME`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` variables before using them.

## Quick Demonstration
---
Here is a [Google Colab Notebook](https://colab.research.google.com/github/HaLU-9000/IncepTIS/blob/main/demo.ipynb) for demonstration of training, inference and design with toy data and light-weight model for understanding IncepTIS.

## Reproducing the Results

> ***Note:*** *Model weights and the training dataset are not yet released. The instructions below are the commands of the publication runs, ready to use once those resources are available.*

Every command in this section expects the offline Hugging Face cache prepared under [Model Training](#model-training):

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### Training Data Format

One flat directory of comma-separated text files named `<transcript>_condition<c>_rep<r>.txt`. All files sharing a `<transcript>` prefix — the text before the first underscore, so transcript identifiers must not contain underscores — stay in the same fold. One row per nucleotide:

```text title="example"
position,base,codon,condition,cosine similarity,normalized_similarity,length coverage,peak_mask
0,G,GAG,0,0.0,0.0,1,False
```

| Column | Used by training | Meaning |
| --- | --- | --- |
| `base` | yes | A, C, G or T; one token per base, padded or truncated to 32,768 positions |
| `condition` | yes | 0 normal, 1 stress; added to the input as a constant channel |
| `cosine similarity` | yes | regression target of the MSE head |
| `peak_mask` | yes | target of the BCE head: the codon of the translation-initiation site at every position whose `cosine similarity` exceeds 0.2, `False` elsewhere |
| `position`, `codon`, `normalized_similarity`, `length coverage` | no | we tried these features while building this model, but not used anymore. carried along |

Note that two column names contain spaces.

### Model Training

To reproduce training process of IncepTIS, install `InstaDeepAI/NTv3_100M_pre` with command below.

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=0
uv run hf auth login  # NTv3_100M_pre is gated: request access on its Hugging Face page first
uv run hf download InstaDeepAI/NTv3_100M_pre --revision 5c685dca15891f5c5b80e0c930e23b87a217e441
uv run hf download InstaDeepAI/ntv3_base_model --revision 0ecff3637f0d3ba5b686d1095083218157c2ca34
mkdir -p "$HF_HOME"/hub/models--InstaDeepAI--{NTv3_100M_pre,ntv3_base_model}/refs
printf %s 5c685dca15891f5c5b80e0c930e23b87a217e441 > "$HF_HOME/hub/models--InstaDeepAI--NTv3_100M_pre/refs/main"
printf %s 0ecff3637f0d3ba5b686d1095083218157c2ca34 > "$HF_HOME/hub/models--InstaDeepAI--ntv3_base_model/refs/main"
```

`examples/train_demo.py` trains one fold with the published B1 settings: context 32,768 tokens, TF32, global batch 16 (4 per GPU across 4 H100), 20 epochs, MSE 0.5 + BCE 0.5, learning rate 1e-3 on a cosine schedule with 100 warmup steps and a 1e-6 minimum, seed 524, 10 transcript folds, one checkpoint per epoch. `--help` lists the overrides; results of fold `N` are written to `<output-dir>/fold_N/` in the layout shown in the demonstration.　<!-- これはReproducing the result ではないので消す。　1gpu ver, 4 gpu ver の再現用コードのみ簡潔に示す -->

One GPU, the ten folds one after another:

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for fold in $(seq 1 10); do
  uv run python examples/train_demo.py \
    --input-dir /path/to/training_files --output-dir /path/to/results \
    --fold "$fold" --epochs 20 --batch-size 2
done
```

Four GPUs for one fold, where batch 4 per GPU restores the B1 global batch of 16:

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 examples/train_demo.py \
  --input-dir /path/to/training_files --output-dir /path/to/results \
  --fold 1 --epochs 20 --batch-size 4 --save-every-epochs 1
```

The training set of the publication was 62,106 files from 15,531 transcripts, and each fold was trained on 4 × H100 94 GB. 

### GENCODE Transcript Prediction

`scripts/run_gencode_inceptis_raw_scores.py` reads a transcript FASTA, scores every position under both conditions with each fold, and writes one gzipped CSV per shard under `shards/per_base_scores/`: one row per position with the `normal_score_fold_<N>` and `stress_score_fold_<N>` columns of every fold and the ensemble `normal_score_mean`, `normal_score_std`, `stress_score_mean` and `stress_score_std` (population standard deviation). Transcripts longer than 32,768 bases are scored in consecutive non-overlapping windows. Each worker also writes `run_summary.worker_<W>.json` with its arguments, folds, shards and timing.

Place the weights as `private/weights/B1_20ep_10fold/fold_<N>/epoch_0020.pt` (or pass the training `--output-dir` as `--weights-root`), then run one worker per GPU:

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/run_gencode_inceptis_raw_scores.py \
    --fasta gencode.v50.transcripts.fa.gz --species human --gencode-release v50 \
    --weights-root private/weights/B1_20ep_10fold --epoch 20 --folds 1-10 \
    --output-dir outputs/gencode_human_scores --device cuda \
    --shard-size 10000 --max-batch-bases 65536 --max-batch-size 64 \
    --num-workers 4 --worker-index "$i" --resume &
done
wait
```

For mouse, replace the FASTA with `--species mouse --gencode-release vM39`. Shards are not merged.

| Release | FASTA from [the GENCODE FTP site](https://ftp.ebi.ac.uk/pub/databases/gencode/) | Transcripts |  md5 |
| --- | --- | --- | --- |
| human v50 | `gencode.v50.transcripts.fa.gz` | 670,670 | `4d5bd41b247480e12c0b370490f52ddd` |
| mouse vM39 | `gencode.vM39.transcripts.fa.gz` | 481,956 | `9926a56585ecc29f63b61ad425d6db8f` |

### Sequence Design

`inceptis-evolve` runs a genetic algorithm over the mutable positions of a fixed template, scored by the B1 model under both conditions, with ExtraTrees-based proposals, Evo2-generated initial sequences, an optional RNAdegformer stability term and a GIF of the search. `examples/translation_switch_uORF3.jsonc` is the configuration of the published uORF3 run.


#### Installing Evo2

To reproduce our design experiment, install Evo 2 with command below (because we used Evo 2 to generate initial seqences).
```bash
mkdir -p private/wheels
gh release download v0.4.0 --repo HaLU-9000/IncepTIS --pattern "transformer_engine_torch-2.12.0-cp312-cp312-linux_x86_64.whl" --dir private/wheels
echo "d672937b92b3c81ea0b59dd0f20cb2ccf45eb099685a0599ce52b674366fedea  private/wheels/transformer_engine_torch-2.12.0-cp312-cp312-linux_x86_64.whl" | sha256sum -c -
uv pip install --python .venv/bin/python -r requirements.evo2-cu128.txt
uv pip install --python .venv/bin/python --no-deps private/wheels/transformer_engine_torch-2.12.0-cp312-cp312-linux_x86_64.whl
uv pip install --python .venv/bin/python flash-attn==2.8.3 --no-build-isolation
uv pip install --python .venv/bin/python --no-deps -e .
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=0
.venv/bin/hf download arcinstitute/evo2_1b_base --revision 2279e1df422c991037470302360edd40d0d2ea1e
mkdir -p "$HF_HOME/hub/models--arcinstitute--evo2_1b_base/refs"
printf %s 2279e1df422c991037470302360edd40d0d2ea1e > "$HF_HOME/hub/models--arcinstitute--evo2_1b_base/refs/main"
```

> **Warning:** `uv sync` removes packages that are not in the lock file, so do not run it again after this step — use `uv sync --locked --inexact`, which keeps them. `uv run` keeps the Evo2 packages and restores the locked versions of the few packages that the Evo2 requirements pin differently (pillow, huggingface-hub, tokenizers, safetensors); the design runs work with that set.

For the installed package, use `--python ~/inceptis-env/bin/python` in place of `--python .venv/bin/python` and omit the `-e .` line.

#### Run design code

The model is loaded through `inceptis_predict.InceptisPredictor`, which expects `model_fold<N>.pt` under `models.exp038_weights_dir`. Link fold 1 — the fold used for the designs — into that layout, then run:

```bash
mkdir -p private/weights/exp038/fold_1
ln -s ../../B1_20ep_10fold/fold_1/epoch_0020.pt private/weights/exp038/fold_1/model_fold1.pt
uv run inceptis-evolve --config examples/translation_switch_uORF3.jsonc
```

Each run creates `<output_root>/<experiment_name>_<timestamp>/`:

| Output | Contents |
| --- | --- |
| `run_inputs/` | the configuration and its resolved copy |
| `best_sequence.fa`, `best_sequence_predictions.csv` | the selected sequence and its per-base predictions |
| `ga_history.csv`, `ga_population_scores.csv` | best candidate per generation; every candidate of every generation |
| `ga_best_profiles.csv`, `ga_top_profiles.csv` | prediction profiles of the best and top candidates |
| `summary.json` | selected sequence, constraint status, site predictions |
| `top_candidates_with_profile.gif` and a final PNG | with `gif.enabled` |


## Python API

Prediction and sequence design are also available from Python. Both load the NTv3 configuration and tokenizer from the Hugging Face cache, so set `HF_HOME`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` first.

```bash
export HF_HOME="$PWD/private/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### Predict your sequences

```python
from inceptis_predict import InceptisPredictor

predictor = InceptisPredictor("/path/to/weights", device="cuda", folds=10)  # model_fold1.pt ... model_fold10.pt
prediction = predictor.predict("GCCACCATGGAC...", sequence_id="example")
prediction.normal, prediction.stress, prediction.stress_normal  # numpy arrays, one value per base
```

### Design your sequences

```python
from pathlib import Path
from inceptis_evolve import SequenceOptimizer, create_experiment_dir, load_jsonc, seed_everything

config_path = Path("translation_switch_uORF3.jsonc")
output_dir, config = create_experiment_dir(load_jsonc(config_path), config_path)
seed_everything(int(config.get("seed", 0)))
optimizer = SequenceOptimizer(config)
summary = optimizer.write_outputs(optimizer.run())   # outputs under output_dir, summary as in summary.json
```
