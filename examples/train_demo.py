#!/usr/bin/env python3
r"""Train one IncepTIS fold with the B1 configuration on a directory of training files.

The B1 20-epoch run on the cluster set a handful of environment variables and started the
unchanged ``kfold_training.run_kfold_training`` through a small sequential runner. This
driver applies the same configuration from the command line and then calls that
function. Values that differ from the B1 run (number of GPUs, batch size per GPU,
and the epoch or step limits used for quick checks) are explicit arguments.

Single GPU, from a clone of the repository::

    uv run python examples/train_demo.py \
        --input-dir /path/to/training_files \
        --output-dir /path/to/results --fold 1

Four GPUs through torchrun, as in the cluster run (batch 4 per GPU gives the B1 global batch 16)::

    uv run torchrun --standalone --nnodes=1 --nproc-per-node=4 examples/train_demo.py \
        --input-dir /path/to/training_files --output-dir /path/to/results --fold 1 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Environment of the B1 job scripts that kfold_training.py reads directly.
B1_ENVIRONMENT = {
    "LR_SCHEDULER": "cosine",
    "WARMUP_STEPS": "100",
    "MIN_LR": "1e-6",
    "WARMUP_LR_INIT": "1e-6",
    "SAVE_EVERY_STEPS": "0",
    "LOG_EVERY_STEPS": "100",
    "SKIP_FINAL_EVAL": "1",
    "RESUME_CHECKPOINT": "",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "8",
}
# The B1 job scripts also set SAVE_EVERY_EPOCHS=1 (a 1.3 GB checkpoint per epoch); the driver
# keeps only the last epoch unless --save-every-epochs says otherwise.
# Configuration the B1 job scripts applied on top of configs/exp038.json.
B1_MODEL = "NTv3unet"
B1_PRETRAINED = "InstaDeepAI/NTv3_100M_pre"
B1_CONTEXT_LENGTH = 32768
B1_EPOCHS = 20
B1_LEARNING_RATE = 1e-3
B1_BATCH_SIZE_PER_GPU = 4  # on four GPUs, global batch 16
B1_BINARIZATION_LOSS = "bce"
B1_BINARIZATION_WEIGHT = 0.5
B1_PRECISION = "tf32"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="directory of <transcript>_condition{0,1}_rep<N>.txt training files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results are written to <output-dir>/fold_<fold>/",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=range(1, 11),
        metavar="1-10",
        help="transcript fold held out for validation (default 1)",
    )
    parser.add_argument("--epochs", type=int, default=B1_EPOCHS, help=f"B1: {B1_EPOCHS}")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=f"batch size per GPU (default 2 for one RTX 5090; B1: {B1_BATCH_SIZE_PER_GPU} per GPU on four H100)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=B1_LEARNING_RATE, help=f"B1: {B1_LEARNING_RATE:g}"
    )
    parser.add_argument(
        "--precision", choices=("tf32", "bf16"), default=B1_PRECISION, help=f"B1: {B1_PRECISION}"
    )
    parser.add_argument("--pretrained", default=B1_PRETRAINED, help=f"B1: {B1_PRETRAINED}")
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=0,
        help="write a checkpoint every N epochs; 0 keeps only the last epoch (B1: 1)",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="training batches per epoch, 0 for all (quick checks only)",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=0,
        help="validation batches per epoch, 0 for all (quick checks only)",
    )
    return parser.parse_args(argv)


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def configure_precision(kfold_training, torch, precision: str) -> None:
    if precision == "tf32":
        # The B1 run disabled bf16 autocast and let float32 matmuls use TF32.
        kfold_training.torch.amp.autocast = lambda *args, **kwargs: nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def main(argv=None) -> None:
    args = parse_args(argv)
    for name, value in B1_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    os.environ["MAX_TRAIN_STEPS"] = str(args.max_train_steps)
    os.environ["MAX_VAL_STEPS"] = str(args.max_val_steps)
    save_every_epochs = args.save_every_epochs if args.save_every_epochs > 0 else args.epochs
    os.environ["SAVE_EVERY_EPOCHS"] = str(save_every_epochs)
    os.environ.pop("BIN_TARGET_THRESHOLD", None)  # B1 used the peak_mask-derived target
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch
    import torch.distributed as dist

    import kfold_training
    from config import Config
    from utils import set_seed

    configure_precision(kfold_training, torch, args.precision)

    cfg = Config.from_json(str(REPO_ROOT / "configs" / "exp038.json"))
    cfg.base_dir = str(REPO_ROOT)
    cfg.input_dir = str(args.input_dir.resolve())
    fold_dir = args.output_dir.resolve() / f"fold_{args.fold}"
    cfg.result_dir = str(fold_dir)
    cfg.model = B1_MODEL
    cfg.pretrained_model_name = args.pretrained
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.context_length = B1_CONTEXT_LENGTH
    cfg.n_splits = 10
    cfg.stop_splits_at = 1
    cfg.loss.binarization.kind = B1_BINARIZATION_LOSS
    cfg.loss.binarization.weight = B1_BINARIZATION_WEIGHT
    cfg.__post_init__()

    done_marker = fold_dir / "_DONE"
    if done_marker.exists():
        if is_rank0():
            print(f"skip fold={args.fold}: {done_marker} exists", flush=True)
        return

    # run_kfold_training iterates over the splits it receives; hand it only the requested fold.
    splitter = kfold_training.create_transcript_stratified_splits
    kfold_training.create_transcript_stratified_splits = (
        lambda *a, **kw: [splitter(*a, **kw)[args.fold - 1]]
    )

    set_seed(cfg.seed)
    if is_rank0():
        print(
            json.dumps(
                {
                    "sequential_fold": args.fold,
                    "epochs": cfg.num_epochs,
                    "gpu_batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "context": cfg.context_length,
                    "precision": args.precision,
                    "binarization_loss": cfg.loss.binarization.kind,
                    "binarization_weight": cfg.loss.binarization.weight,
                    "focal_alpha": cfg.loss.binarization.focal_alpha,
                    "focal_gamma": cfg.loss.binarization.focal_gamma,
                    "input": cfg.input_path,
                    "output": cfg.result_path,
                    "save_every_epochs": save_every_epochs,
                    "auto_resume": False,
                    "resume_checkpoint": None,
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                },
                indent=2,
            ),
            flush=True,
        )

    try:
        kfold_training.run_kfold_training(
            cfg, model_name=cfg.model, plot_results=True, positional_config=None
        )
        if is_rank0():
            done_marker.parent.mkdir(parents=True, exist_ok=True)
            done_marker.write_text("done\n")
            print(f"completed fold={args.fold} marker={done_marker}", flush=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
