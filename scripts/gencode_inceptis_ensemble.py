#!/usr/bin/env python3
"""GENCODE input handling and fold-ensemble inference for the B1 IncepTIS model.

This module is the inference part of the program that scored GENCODE human v50 and
mouse vM39 on the cluster; the start-codon and ORF extraction of that program is not
part of the public release. ``run_gencode_inceptis_raw_scores.py`` imports it to write
the per-base scores of every transcript.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inceptis_predict.model import Exp038Model, hf_token
from inceptis_predict.predictor import CONTEXT_LENGTH, PAD_MULTIPLE, InceptisPredictor


DEFAULT_WEIGHTS = ROOT / "private/weights/B1_20ep_10fold"
BACKBONE = "InstaDeepAI/NTv3_100M_pre"


@dataclass(frozen=True)
class Transcript:
    header: str
    transcript_id: str
    gene_id: str
    havana_gene_id: str
    havana_transcript_id: str
    transcript_name: str
    gene_name: str
    transcript_type: str
    sequence: str


@dataclass(frozen=True)
class Chunk:
    record_index: int
    start: int
    length: int


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii")
    return path.open("r", encoding="ascii")


def parse_gencode_header(header: str) -> dict[str, str]:
    fields = header.split("|")
    if fields and fields[-1] == "":
        fields.pop()
    fields += [""] * max(0, 8 - len(fields))
    return {
        "transcript_id": fields[0],
        "gene_id": fields[1],
        "havana_gene_id": fields[2],
        "havana_transcript_id": fields[3],
        "transcript_name": fields[4],
        "gene_name": fields[5],
        "transcript_type": fields[7],
    }


def read_fasta(path: Path, limit: int = 0) -> Iterator[Transcript]:
    header: str | None = None
    sequence_parts: list[str] = []
    emitted = 0
    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield make_transcript(header, sequence_parts, path)
                    emitted += 1
                    if limit and emitted >= limit:
                        return
                header = line[1:]
                sequence_parts = []
            elif header is None:
                raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
            else:
                sequence_parts.append(line)
    if header is not None and (not limit or emitted < limit):
        yield make_transcript(header, sequence_parts, path)


def make_transcript(header: str, parts: list[str], path: Path) -> Transcript:
    sequence = "".join(parts).upper().replace("U", "T")
    invalid = sorted(set(sequence) - set("ACGTN"))
    if not sequence:
        raise ValueError(f"{path}: empty sequence for {header}")
    if invalid:
        raise ValueError(f"{path}: unsupported bases {''.join(invalid)} in {header}")
    metadata = parse_gencode_header(header)
    return Transcript(header=header, sequence=sequence, **metadata)


def shards(records: Iterable[Transcript], size: int) -> Iterator[list[Transcript]]:
    shard: list[Transcript] = []
    for record in records:
        shard.append(record)
        if len(shard) == size:
            yield shard
            shard = []
    if shard:
        yield shard


def parse_folds(value: str) -> list[int]:
    folds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            folds.extend(range(start, stop + 1))
        elif part:
            folds.append(int(part))
    if not folds or len(folds) != len(set(folds)) or any(f < 1 or f > 10 for f in folds):
        raise ValueError(f"invalid folds: {value}")
    return folds


def find_checkpoints(root: Path, folds: list[int], epoch: int) -> list[tuple[int, Path]]:
    checkpoints = []
    for fold in folds:
        candidates = (
            root / f"fold_{fold}" / f"epoch_{epoch:04d}.pt",
            root
            / f"fold_{fold}/checkpoints/NTv3unet/fold_1"
            / f"epoch_{epoch:04d}.pt",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"B1 fold {fold} epoch {epoch}: tried {candidates}")
        checkpoints.append((fold, path))
    return checkpoints


class EnsembleRunner:
    def __init__(
        self,
        checkpoints: list[tuple[int, Path]],
        device: str,
        max_batch_bases: int,
        max_batch_size: int,
        log_every_batches: int,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.checkpoints = checkpoints
        self.max_batch_bases = max_batch_bases
        self.max_batch_size = max_batch_size
        self.log_every_batches = log_every_batches
        self.tokenizer = AutoTokenizer.from_pretrained(
            BACKBONE, trust_remote_code=True, token=hf_token()
        )
        self.model = Exp038Model(BACKBONE).to(self.device).eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    def predict(
        self,
        records: list[Transcript],
        retain_fold_scores: bool = False,
        aggregate_scores: bool = True,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[list[np.ndarray]],
        list[list[np.ndarray]],
        dict,
    ]:
        normal = (
            [np.zeros(len(record.sequence), dtype=np.float32) for record in records]
            if aggregate_scores
            else []
        )
        stress = (
            [np.zeros(len(record.sequence), dtype=np.float32) for record in records]
            if aggregate_scores
            else []
        )
        normal_by_fold: list[list[np.ndarray]] = []
        stress_by_fold: list[list[np.ndarray]] = []
        chunks = self._chunks(records)
        fold_metrics = []
        for fold, checkpoint_path in self.checkpoints:
            load_start = time.perf_counter()
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True, mmap=True
            )
            if checkpoint.get("epoch") not in (None, int(checkpoint_path.stem.split("_")[-1])):
                raise RuntimeError(f"unexpected epoch metadata in {checkpoint_path}")
            state = InceptisPredictor._state_dict(checkpoint)
            InceptisPredictor._load_state_dict(self.model, state)
            del checkpoint, state
            self.model.eval()
            load_seconds = time.perf_counter() - load_start

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            inference_start = time.perf_counter()
            batch_count = 0
            processed_bases = 0
            fold_normal = [
                np.zeros(len(record.sequence), dtype=np.float32) for record in records
            ]
            fold_stress = [
                np.zeros(len(record.sequence), dtype=np.float32) for record in records
            ]
            for batch in self._batches(chunks):
                batch_count += self._predict_batch(
                    records, batch, fold_normal, fold_stress
                )
                processed_bases += sum(chunk.length for chunk in batch)
                if self.log_every_batches and batch_count % self.log_every_batches == 0:
                    elapsed = time.perf_counter() - inference_start
                    print(
                        f"fold={fold} batches={batch_count} bases={processed_bases} "
                        f"bases_per_second={processed_bases / max(elapsed, 1e-9):.1f}",
                        flush=True,
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_seconds = time.perf_counter() - inference_start
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / 1024**3
                if self.device.type == "cuda"
                else 0.0
            )
            fold_metrics.append(
                {
                    "fold": fold,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "load_seconds": load_seconds,
                    "inference_seconds": inference_seconds,
                    "batches": batch_count,
                    "bases": processed_bases,
                    "bases_per_second": processed_bases / max(inference_seconds, 1e-9),
                    "peak_gpu_memory_gib": peak_gib,
                }
            )
            print(
                f"fold={fold} complete seconds={inference_seconds:.3f} "
                f"bases_per_second={processed_bases / max(inference_seconds, 1e-9):.1f} "
                f"peak_gpu_memory_gib={peak_gib:.3f}",
                flush=True,
            )

            if aggregate_scores:
                for mean_values, fold_values in zip(normal, fold_normal):
                    mean_values += fold_values
                for mean_values, fold_values in zip(stress, fold_stress):
                    mean_values += fold_values
            if retain_fold_scores:
                normal_by_fold.append(fold_normal)
                stress_by_fold.append(fold_stress)

        if aggregate_scores:
            scale = np.float32(1.0 / len(self.checkpoints))
            for values in normal:
                values *= scale
            for values in stress:
                values *= scale
        return normal, stress, normal_by_fold, stress_by_fold, {"folds": fold_metrics}

    @staticmethod
    def _chunks(records: list[Transcript]) -> list[Chunk]:
        chunks = []
        for record_index, record in enumerate(records):
            for start in range(0, len(record.sequence), CONTEXT_LENGTH):
                chunks.append(
                    Chunk(record_index, start, min(CONTEXT_LENGTH, len(record.sequence) - start))
                )
        return sorted(chunks, key=lambda chunk: chunk.length)

    def _batches(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        batch: list[Chunk] = []
        padded_length = 0
        for chunk in chunks:
            candidate_length = max(padded_length, chunk.length)
            candidate_size = len(batch) + 1
            too_large = candidate_length * candidate_size > self.max_batch_bases
            too_many = candidate_size > self.max_batch_size
            if batch and (too_large or too_many):
                yield batch
                batch = []
                padded_length = 0
            batch.append(chunk)
            padded_length = max(padded_length, chunk.length)
        if batch:
            yield batch

    def _predict_batch(
        self,
        records: list[Transcript],
        batch: list[Chunk],
        normal: list[np.ndarray],
        stress: list[np.ndarray],
    ) -> int:
        try:
            predictions = self._forward(records, batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(batch) // 2
            return self._predict_batch(records, batch[:middle], normal, stress) + self._predict_batch(
                records, batch[middle:], normal, stress
            )
        for row, chunk in enumerate(batch):
            stop = chunk.start + chunk.length
            normal[chunk.record_index][chunk.start:stop] += predictions[0][row, : chunk.length]
            stress[chunk.record_index][chunk.start:stop] += predictions[1][row, : chunk.length]
        return 1

    def _forward(
        self, records: list[Transcript], batch: list[Chunk]
    ) -> tuple[np.ndarray, np.ndarray]:
        sequences = [
            records[chunk.record_index].sequence[chunk.start : chunk.start + chunk.length]
            for chunk in batch
        ]
        encoded = self.tokenizer(sequences, padding=True, return_tensors="pt")
        tokens = encoded["input_ids"]
        longest = max(chunk.length for chunk in batch)
        if tokens.shape[1] != longest:
            raise RuntimeError(
                f"tokenizer produced length {tokens.shape[1]} for {longest} input bases"
            )
        padded = ((longest + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        if padded > longest:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise RuntimeError("backbone tokenizer has no pad token")
            padding = torch.full(
                (tokens.shape[0], padded - longest), pad_id, dtype=tokens.dtype
            )
            tokens = torch.cat((tokens, padding), dim=1)
        lengths = torch.tensor([chunk.length for chunk in batch], dtype=torch.long)
        mask = torch.arange(padded).unsqueeze(0) < lengths.unsqueeze(1)
        tokens = tokens.repeat_interleave(2, dim=0).to(self.device)
        mask = mask.repeat_interleave(2, dim=0).to(self.device)
        conditions = torch.zeros((len(batch) * 2, padded, 1), device=self.device)
        conditions[1::2] = 1.0
        with torch.inference_mode():
            output = self.model(tokens=tokens, conditions=conditions, mask=mask)["regression"]
        values = output[:, :longest].float().cpu().numpy()
        return values[0::2], values[1::2]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)
