"""Fixed-segment DNA design utilities."""

from __future__ import annotations

import numpy as np


BASES = np.array(list("ACGT"))


def clean_dna(seq: str) -> str:
    """Normalize a sequence to A/C/G/T, converting U to T and dropping others."""
    return "".join(base for base in str(seq).upper().replace("U", "T") if base in "ACGT")


class FixedDesign:
    """A fixed/mutable DNA design with named target sites."""

    def __init__(self, cfg: dict):
        self.length = int(cfg["length"])
        self.sites = cfg["sites"]
        self.site_by_name = {site["name"]: site for site in self.sites}
        self.uorf_names = [site["name"] for site in self.sites if site.get("group") == "uorf"]
        luc = [site["name"] for site in self.sites if site.get("group") == "luciferase"]
        if len(luc) != 1:
            raise ValueError("Exactly one site with group='luciferase' is required")
        self.luc_name = luc[0]
        self.site_indices = {site["name"]: int(site["position"]) - 1 for site in self.sites}

        self.template = ["N"] * self.length
        self.fixed = np.zeros(self.length, dtype=bool)
        self.labels = ["mutable"] * self.length
        for segment in cfg["fixed_segments"]:
            label = segment["label"]
            start = int(segment["start"]) - 1
            seq = clean_dna(segment["sequence"])
            end = start + len(seq)
            if start < 0 or end > self.length:
                raise ValueError(f"fixed segment {label} does not fit")
            for i, base in enumerate(seq, start):
                if self.fixed[i] and self.template[i] != base:
                    raise ValueError(f"conflicting fixed base at position {i + 1}")
                self.template[i] = base
                self.fixed[i] = True
                self.labels[i] = label
        self.mutable_positions = np.flatnonzero(~self.fixed)
        self.fixed_positions = np.flatnonzero(self.fixed)

    def overlay_fixed(self, seq: str) -> str:
        """Apply fixed positions to a raw sequence, padding or trimming to length."""
        seq = clean_dna(seq)
        if len(seq) < self.length:
            seq += "".join(np.random.choice(BASES, size=self.length - len(seq)))
        chars = list(seq[: self.length])
        for pos in self.fixed_positions:
            chars[int(pos)] = self.template[int(pos)]
        return "".join(chars)

    def random_sequence(self, rng=np.random) -> str:
        """Generate a sequence that respects fixed positions."""
        chars = self.template.copy()
        for pos, base in zip(self.mutable_positions, rng.choice(BASES, size=len(self.mutable_positions))):
            chars[int(pos)] = str(base)
        return "".join(chars)

    def clean_sequence(self, seq: str) -> str:
        """Validate length and fixed positions."""
        seq = clean_dna(seq)
        if len(seq) != self.length:
            raise ValueError(f"Expected length {self.length}, got {len(seq)}")
        for pos in self.fixed_positions:
            if seq[int(pos)] != self.template[int(pos)]:
                raise ValueError(f"fixed position {int(pos) + 1} changed")
        return seq

    def local_mutable_mask(self, radius: int) -> np.ndarray:
        """Mutable positions near target sites, used for elevated mutation rate."""
        mask = np.zeros(self.length, dtype=bool)
        for idx in self.site_indices.values():
            mask[max(0, idx - radius): min(self.length, idx + radius + 3)] = True
        return mask & ~self.fixed

    def internal_mask(self, radius: int = 4) -> np.ndarray:
        """Positions not near explicitly requested TIS sites."""
        mask = np.ones(self.length, dtype=bool)
        for idx in self.site_indices.values():
            mask[max(0, idx - radius): min(self.length, idx + radius + 1)] = False
        return mask
