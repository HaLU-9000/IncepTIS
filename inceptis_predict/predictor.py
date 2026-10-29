"""Loading and ensemble inference for exp038."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

from .model import Exp038Model, hf_token

CONTEXT_LENGTH = 32768
PAD_MULTIPLE = 128


@dataclass(frozen=True)
class Prediction:
    sequence_id: str
    sequence: str
    normal: np.ndarray
    stress: np.ndarray

    @property
    def stress_normal(self) -> np.ndarray:
        return self.stress - self.normal


class InceptisPredictor:
    """Memory-conscious ten-fold exp038 ensemble predictor."""

    def __init__(self, weights_dir, device="auto",
                 backbone="InstaDeepAI/NTv3_100M_pre", folds=10):
        self.weights_dir = Path(weights_dir)
        self.folds = int(folds)
        if self.folds < 1:
            raise ValueError("folds must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backbone_name = backbone
        token = hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=True, token=token)
        self.checkpoints = [self._find_checkpoint(i) for i in range(1, self.folds + 1)]

    def _find_checkpoint(self, fold):
        candidates = (
            self.weights_dir / f"model_fold{fold}.pt",
            self.weights_dir / "NTv3unet" / f"model_fold{fold}.pt",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"fold {fold} checkpoint not found under {self.weights_dir}; "
            f"expected model_fold{fold}.pt")

    @staticmethod
    def _clean_sequence(sequence):
        sequence = "".join(str(sequence).split()).upper().replace("U", "T")
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(f"unsupported sequence characters: {''.join(invalid)}")
        if not sequence:
            raise ValueError("sequence is empty")
        return sequence

    @staticmethod
    def _state_dict(checkpoint):
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        return {key.replace("backbone._orig_mod.", "backbone."): value
                for key, value in state.items()}

    @staticmethod
    def _load_state_dict(model, state):
        allowed = ("rotary_embedding.cos_cached", "rotary_embedding.sin_cached")
        # NTv3 resizes these derived rotary caches to the latest input length.
        # Loading a later fold after inference can otherwise cause a harmless
        # cache-shape mismatch even though all trainable parameters agree.
        state = {key: value for key, value in state.items()
                 if not key.endswith(allowed)}
        incompatible = model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys
                   if not key.endswith(allowed)]
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.endswith(allowed)]
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint does not match exp038 architecture: "
                f"missing={missing}, unexpected={unexpected}")

    def predict(self, sequence, sequence_id="sequence"):
        sequence = self._clean_sequence(sequence)
        normal_sum = np.zeros(len(sequence), dtype=np.float64)
        stress_sum = np.zeros(len(sequence), dtype=np.float64)
        model = Exp038Model(self.backbone_name)

        for checkpoint_path in self.checkpoints:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
            self._load_state_dict(model, self._state_dict(checkpoint))
            del checkpoint
            model.to(self.device).eval()
            for start in range(0, len(sequence), CONTEXT_LENGTH):
                chunk = sequence[start:start + CONTEXT_LENGTH]
                normal, stress = self._predict_chunk(model, chunk)
                stop = start + len(chunk)
                normal_sum[start:stop] += normal
                stress_sum[start:stop] += stress
            model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        scale = 1.0 / len(self.checkpoints)
        return Prediction(str(sequence_id), sequence,
                          (normal_sum * scale).astype(np.float32),
                          (stress_sum * scale).astype(np.float32))

    def predict_many(self, records: Iterable[tuple[str, str]]):
        for sequence_id, sequence in records:
            yield self.predict(sequence, sequence_id)

    def _predict_chunk(self, model, sequence):
        encoded = self.tokenizer(sequence, return_tensors="pt")
        tokens = encoded["input_ids"]
        original = tokens.shape[1]
        if original != len(sequence):
            raise RuntimeError(
                f"tokenizer produced {original} tokens for {len(sequence)} bases; "
                "the exp038 model requires one token per base")
        padded = ((original + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        if padded > original:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise RuntimeError("backbone tokenizer has no pad_token_id")
            tokens = torch.cat([tokens, torch.full((1, padded - original), pad_id,
                                                    dtype=tokens.dtype)], dim=1)
        tokens = tokens.repeat(2, 1).to(self.device)
        conditions = torch.zeros((2, padded, 1), device=self.device)
        conditions[1] = 1.0
        mask = torch.zeros((2, padded), device=self.device)
        mask[:, :original] = 1.0
        with torch.inference_mode():
            output = model(tokens=tokens, conditions=conditions, mask=mask)["regression"]
        values = output[:, :original].float().cpu().numpy()
        return values[0], values[1]
