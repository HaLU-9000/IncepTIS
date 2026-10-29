"""Model adapter classes used by the optimizer.

The package keeps model weights outside the wheel. These adapters import heavy
model packages lazily so the core package can still be installed without them.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .design import clean_dna


RNADEG_TARGET_COLUMNS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


class BatchExp038Scorer:
    """Batch scorer for the exp038 IncepTIS model via `inceptis-predict`."""

    def __init__(self, weights_dir: str, design, device: str, backbone: str, folds: int = 1):
        import torch
        from inceptis_predict.model import Exp038Model
        from inceptis_predict.predictor import InceptisPredictor, PAD_MULTIPLE

        self.torch = torch
        self.pad_multiple = PAD_MULTIPLE
        self.design = design
        self.device = torch.device(device)
        self.predictor = InceptisPredictor(weights_dir, device="cpu", backbone=backbone, folds=folds)
        self.model = Exp038Model(backbone)
        state_sum = None
        for checkpoint_path in self.predictor.checkpoints:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
            state = self.predictor._state_dict(ckpt)
            if state_sum is None:
                state_sum = {k: v.detach().float().clone() for k, v in state.items()}
            else:
                for k, v in state.items():
                    state_sum[k] += v.detach().float()
            del ckpt
        scale = 1.0 / len(self.predictor.checkpoints)
        self.predictor._load_state_dict(self.model, {k: v * scale for k, v in state_sum.items()})
        self.model.to(self.device).eval()

    def predict_batch(self, sequences: list[str], batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        sequences = [self.design.clean_sequence(seq) for seq in sequences]
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(sequences), batch_size):
            out.extend(self._predict_chunk(sequences[start:start + batch_size]))
        return out

    def _predict_chunk(self, sequences: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        torch = self.torch
        encoded = self.predictor.tokenizer(sequences, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"]
        lengths = [len(seq) for seq in sequences]
        padded = ((tokens.shape[1] + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        if padded > tokens.shape[1]:
            pad_id = self.predictor.tokenizer.pad_token_id
            tokens = torch.cat([tokens, torch.full((tokens.shape[0], padded - tokens.shape[1]), pad_id, dtype=tokens.dtype)], dim=1)
        repeated = tokens.repeat_interleave(2, dim=0).to(self.device)
        conditions = torch.zeros((repeated.shape[0], padded, 1), device=self.device)
        conditions[1::2] = 1.0
        mask = torch.zeros((repeated.shape[0], padded), device=self.device)
        for i, n in enumerate(lengths):
            mask[2 * i:2 * i + 2, :n] = 1.0
        with torch.inference_mode():
            pred = self.model(tokens=repeated, conditions=conditions, mask=mask)["regression"]
        values = pred.float().cpu().numpy()
        normal, stress = values[0::2], values[1::2]
        return [(normal[i, :n].copy(), stress[i, :n].copy()) for i, n in enumerate(lengths)]


class Evo2Scorer:
    """Lazy Evo2 wrapper used for generation and naturalness scoring."""

    def __init__(self, model_name: str):
        from evo2 import Evo2

        self.model = Evo2(model_name)

    def score_sequences(self, sequences: list[str], batch_size: int) -> np.ndarray:
        scores: list[float] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            scores.extend(
                self.model.score_sequences(
                    chunk,
                    batch_size=len(chunk),
                    prepend_bos=False,
                    reduce_method="mean",
                    average_reverse_complement=False,
                )
            )
        return np.asarray(scores, dtype=float)

    def generate(self, prompt: str, n_tokens: int, n: int, top_k: int = 4, temperature: float = 1.0) -> list[str]:
        seqs: list[str] = []
        while len(seqs) < n:
            current = min(10, n - len(seqs))
            output = self.model.generate(
                prompt_seqs=[prompt] * current,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                batched=True,
                cached_generation=True,
                verbose=0,
            )
            generated = output.sequences if hasattr(output, "sequences") else output[0]
            seqs.extend(clean_dna(seq) for seq in generated)
        return seqs[:n]


class RNAdegformerStabilityScorer:
    """RNAdegformer wrapper that scores candidate stability for top candidates."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.cache: dict[str, dict] = {}
        self.model = None
        if not self.enabled:
            return

        root = Path(cfg.get("project_root", ".")).resolve()
        self.repo_dir = (root / cfg.get("webapp_dir", "third_party/RNAdegformer-Webapp")).resolve()
        self.arnie_src = (root / cfg.get("arnie_src", "third_party/arnie/src")).resolve()
        self.weights_dir = (root / cfg.get("weights_dir", "third_party/RNAdegformer-Webapp/RNA_Inference/best_weights")).resolve()
        self.targets = cfg.get("target_columns", RNADEG_TARGET_COLUMNS)
        self.folds = cfg.get("folds", "all")
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="tis_rnadeg_")
        runtime_dir = Path(self._runtime_dir.name)
        self.tmp_dir = runtime_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.rnasoft_dir = runtime_dir / "rnasoft"
        self.rnasoft_dir.symlink_to(self.repo_dir / "MultiRNAFold", target_is_directory=True)
        self.arnie_conf = runtime_dir / "arnie.conf"

        for path in (self.arnie_src, self.repo_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.environ["ARNIEFILE"] = str(self.arnie_conf)
        self._write_arnie_conf()

        cwd = Path.cwd()
        try:
            os.chdir(self.repo_dir)
            from RNA_Inference.inference import RNA_Inference
        finally:
            os.chdir(cwd)

        self.model = RNA_Inference()
        self.model.load_models(str(self.weights_dir))
        if self.folds != "all":
            self.model.models = self.model.models[: int(self.folds)]

    def _write_arnie_conf(self) -> None:
        self.arnie_conf.write_text(
            f"TMP: {self.tmp_dir}\nrnasoft: {self.rnasoft_dir}\n",
            encoding="utf-8",
        )

    @staticmethod
    def dna_to_rna(seq: str) -> str:
        return clean_dna(seq).replace("T", "U")

    def score_one(self, sequence: str) -> dict:
        if not self.enabled:
            return {}
        sequence = clean_dna(sequence)
        if sequence in self.cache:
            return self.cache[sequence]
        cwd = Path.cwd()
        try:
            os.chdir(self.tmp_dir)
            pred, _, _ = self.model.predict(self.dna_to_rna(sequence))
        finally:
            os.chdir(cwd)
        pred = np.asarray(pred, dtype=float)
        target_idx = [RNADEG_TARGET_COLUMNS.index(target) for target in self.targets]
        target_vals = pred[:, target_idx]
        mean_degradation = float(np.nanmean(target_vals))
        result = {
            "rnadegformer_mean_degradation": mean_degradation,
            "rnadegformer_stability_score": -mean_degradation,
        }
        for col, val in zip(RNADEG_TARGET_COLUMNS, np.nanmean(pred, axis=0)):
            result[f"rnadegformer_mean_{col}"] = float(val)
        self.cache[sequence] = result
        return result

    def score_many(self, sequences: list[str]) -> list[dict]:
        return [self.score_one(seq) for seq in sequences]
