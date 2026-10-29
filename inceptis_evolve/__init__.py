"""Sequence evolution tools for translation-initiation design."""

from .design import FixedDesign, clean_dna
from .jsonc import load_jsonc, strip_jsonc

__all__ = [
    "FixedDesign",
    "SequenceOptimizer",
    "clean_dna",
    "create_experiment_dir",
    "load_jsonc",
    "seed_everything",
    "strip_jsonc",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"SequenceOptimizer", "create_experiment_dir", "seed_everything"}:
        from . import optimizer

        return getattr(optimizer, name)
    raise AttributeError(name)
