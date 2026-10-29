#!/usr/bin/env python3
"""Write per-base IncepTIS scores for every requested fold and condition."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from gencode_inceptis_ensemble import (
    DEFAULT_WEIGHTS,
    EnsembleRunner,
    find_checkpoints,
    parse_folds,
    read_fasta,
    shards,
    write_json,
)


def score_columns(folds: list[int]) -> list[str]:
    columns = ["transcript_id", "position_1based"]
    for fold in folds:
        columns.extend((f"normal_score_fold_{fold}", f"stress_score_fold_{fold}"))
    columns.extend(
        ("normal_score_mean", "normal_score_std", "stress_score_mean", "stress_score_std")
    )
    return columns


def write_score_csv(
    path: Path,
    records: list,
    folds: list[int],
    normal_by_fold: list[list[np.ndarray]],
    stress_by_fold: list[list[np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with gzip.open(
        partial, "wt", encoding="ascii", newline="", compresslevel=1
    ) as handle:
        handle.write(",".join(score_columns(folds)) + "\n")
        for record_index, record in enumerate(records):
            normal = np.stack(
                [values[record_index] for values in normal_by_fold], axis=0
            )
            stress = np.stack(
                [values[record_index] for values in stress_by_fold], axis=0
            )
            normal_mean = normal.mean(axis=0)
            normal_std = normal.std(axis=0, ddof=0)
            stress_mean = stress.mean(axis=0)
            stress_std = stress.std(axis=0, ddof=0)
            for position in range(len(record.sequence)):
                fold_values = []
                for fold_index in range(len(folds)):
                    fold_values.extend(
                        (
                            f"{normal[fold_index, position]:.8g}",
                            f"{stress[fold_index, position]:.8g}",
                        )
                    )
                summary_values = (
                    f"{normal_mean[position]:.8g}",
                    f"{normal_std[position]:.8g}",
                    f"{stress_mean[position]:.8g}",
                    f"{stress_std[position]:.8g}",
                )
                handle.write(
                    f"{record.transcript_id},{position + 1},"
                    + ",".join((*fold_values, *summary_values))
                    + "\n"
                )
    partial.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--species", required=True, choices=("human", "mouse"))
    parser.add_argument("--gencode-release", required=True)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--folds", default="1-10")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--max-batch-bases", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    folds = parse_folds(args.folds)
    checkpoints = find_checkpoints(args.weights_root, folds, args.epoch)
    runner: EnsembleRunner | None = None
    run_start = time.perf_counter()
    completed = []

    for shard_number, records in enumerate(
        shards(read_fasta(args.fasta, args.limit), args.shard_size), 1
    ):
        if (shard_number - 1) % args.num_workers != args.worker_index:
            continue
        shard_name = f"shard_{shard_number:06d}"
        score_path = args.output_dir / "shards/per_base_scores" / f"{shard_name}.csv.gz"
        metric_path = args.output_dir / "shards/metrics" / f"{shard_name}.json"
        if (
            args.resume
            and score_path.is_file()
            and metric_path.is_file()
        ):
            completed.append(json.loads(metric_path.read_text(encoding="utf-8")))
            print(f"shard={shard_number} skipped=complete", flush=True)
            continue

        if runner is None:
            runner = EnsembleRunner(
                checkpoints,
                args.device,
                args.max_batch_bases,
                args.max_batch_size,
                args.log_every_batches,
            )
        shard_start = time.perf_counter()
        _, _, normal_by_fold, stress_by_fold, prediction_metrics = runner.predict(
            records, retain_fold_scores=True, aggregate_scores=False
        )
        write_score_csv(score_path, records, folds, normal_by_fold, stress_by_fold)
        metric = {
            "shard": shard_number,
            "transcripts": len(records),
            "bases": sum(len(record.sequence) for record in records),
            "folds": folds,
            "conditions": ["normal", "stress"],
            "prediction": prediction_metrics,
            "wall_seconds": time.perf_counter() - shard_start,
        }
        write_json(metric_path, metric)
        completed.append(metric)
        print(
            f"shard={shard_number} complete transcripts={metric['transcripts']} "
            f"bases={metric['bases']} seconds={metric['wall_seconds']:.3f}",
            flush=True,
        )

    summary = {
        "input": {
            "fasta": str(args.fasta.resolve()),
            "species": args.species,
            "gencode_release": args.gencode_release,
        },
        "model": {
            "name": "B1 longread-reference hybrid epoch 20",
            "epoch": args.epoch,
            "folds": folds,
            "conditions": ["normal", "stress"],
        },
        "score_format": {
            "format": "gzip-compressed CSV",
            "row": "one transcript nucleotide position",
            "position": "one-based transcript position",
            "fold_score_dtype": "float32 formatted with 8 significant digits",
            "standard_deviation": "population standard deviation (ddof=0)",
            "columns": score_columns(folds),
        },
        "counts": {
            "transcripts": sum(item["transcripts"] for item in completed),
            "bases": sum(item["bases"] for item in completed),
            "completed_shards": len(completed),
        },
        "wall_seconds": time.perf_counter() - run_start,
    }
    write_json(
        args.output_dir / f"run_summary.worker_{args.worker_index:02d}.json", summary
    )
    print(json.dumps(summary["counts"]), flush=True)


if __name__ == "__main__":
    main()
