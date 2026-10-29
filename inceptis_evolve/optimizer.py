"""Config-driven GA/BO sequence optimizer."""

from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .design import BASES, FixedDesign, clean_dna
from .scorers import BatchExp038Scorer, Evo2Scorer, RNAdegformerStabilityScorer

try:
    from sklearn.ensemble import ExtraTreesRegressor
except Exception:
    ExtraTreesRegressor = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def create_experiment_dir(cfg: dict, config_path: Path) -> tuple[Path, dict]:
    root_value = cfg.get("output_root")
    if root_value is None:
        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return out, cfg

    root = Path(root_value)
    name = cfg.get("experiment_name", "translation_switch")
    if cfg.get("timestamp_experiment_dir", True):
        name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = root / name
    suffix = 1
    base_out = out
    while out.exists():
        suffix += 1
        out = Path(f"{base_out}_{suffix:02d}")
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(cfg)
    cfg["output_dir"] = str(out)
    metadata = out / "run_inputs"
    metadata.mkdir(exist_ok=True)
    shutil.copy2(config_path, metadata / config_path.name)
    for path_value in cfg.get("copy_to_experiment", []):
        src = Path(path_value)
        if src.exists() and src.is_file():
            shutil.copy2(src, metadata / src.name)
    (metadata / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out, cfg


class SequenceOptimizer:
    """Genetic optimizer with optional Evo2, RNAdegformer, and BO terms."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.design = FixedDesign(cfg)
        self.ga = cfg["ga"]
        self.bo = cfg.get("bo", {"enabled": False})
        self.evo_cfg = cfg.get("evo2_constraint", {})
        self.constraint_mode = cfg.get("naturalness_constraint", "none")
        self.constraints = cfg["success_constraints"]
        self.weights = cfg["objective_weights"]
        self.local_mask = self.design.local_mutable_mask(int(self.ga["local_mutation_radius"]))
        self.internal_mask = self.design.internal_mask()

        device = cfg["models"].get("device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        self.exp038 = BatchExp038Scorer(
            cfg["models"]["exp038_weights_dir"],
            self.design,
            device,
            cfg["models"].get("exp038_backbone", "InstaDeepAI/NTv3_100M_pre"),
            int(cfg["models"].get("exp038_folds", 1)),
        )
        self.evo2 = None
        if self.constraint_mode == "evo2" or int(self.evo_cfg.get("initial_seed_count", 0)) > 0:
            self.evo2 = Evo2Scorer(cfg["models"]["evo2_model"])
        self.stability = RNAdegformerStabilityScorer(cfg.get("stability_model", {}))
        self.bo_sequences: list[str] = []
        self.bo_scores: list[float] = []

    @staticmethod
    def safe_key(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")

    def metric_value(self, summary: dict, spec: dict) -> float:
        if "metric" in spec:
            return float(summary[spec["metric"]])
        if "condition" in spec and "site" in spec:
            return float(summary[f"{spec['condition']}_{self.safe_key(spec['site'])}"])
        if "difference" in spec and "site" in spec:
            return float(summary[f"{spec['difference']}_{self.safe_key(spec['site'])}"])
        raise ValueError(f"constraint/objective needs metric or site+condition/difference: {spec}")

    def evo2_score(self, sequences: list[str]) -> np.ndarray:
        if self.evo2 is None:
            return np.full(len(sequences), np.nan)
        return self.evo2.score_sequences(sequences, int(self.evo_cfg.get("batch_size", len(sequences))))

    def evo2_generate_raw(self, n: int) -> list[str]:
        if n <= 0:
            return []
        if self.evo2 is None:
            raise RuntimeError("Evo2 initial_seed_count > 0 requires Evo2 to be installed and configured")
        raw = self.evo2.generate(
            prompt=self.evo_cfg.get("prompt", "ACGT"),
            n_tokens=self.design.length - len(self.evo_cfg.get("prompt", "ACGT")),
            n=n,
            temperature=float(self.evo_cfg.get("temperature", 1.0)),
            top_k=int(self.evo_cfg.get("top_k", 4)),
        )
        return [clean_dna(seq)[: self.design.length] for seq in raw]

    def estimate_evo2_threshold(self) -> float:
        if self.constraint_mode != "evo2":
            return float("-inf")
        raw = self.evo2_generate_raw(int(self.evo_cfg["raw_baseline_n"]))
        vals = self.evo2_score(raw)
        threshold = float(vals.mean() + float(self.evo_cfg["sd_multiplier"]) * vals.std(ddof=1))
        pd.DataFrame({"condition": "raw_evo2_generated", "index": np.arange(len(raw)), "evo2_mean_logprob": vals, "sequence": raw}).to_csv(
            self.out / "raw_evo2_baseline.csv", index=False
        )
        summary = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "threshold": threshold,
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
        }
        (self.out / "raw_evo2_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return threshold

    def add_stability_scores(self, population: list[str], exp_scores: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, dict[int, dict], float]:
        if not self.stability.enabled:
            return np.full(len(population), np.nan), {}, float("nan")
        stability_scores = np.full(len(population), np.nan)
        stability_records: dict[int, dict] = {}
        topk = int(self.cfg["stability_model"].get("topk_per_generation", self.evo_cfg.get("topk_per_generation", 0)))
        top_pool = np.argsort(exp_scores)[::-1][:topk]
        scored = self.stability.score_many([population[int(i)] for i in top_pool])
        vals = np.array([item["rnadegformer_stability_score"] for item in scored], dtype=float)
        z = (vals - vals.mean()) / vals.std(ddof=1) if len(vals) > 1 and vals.std(ddof=1) > 1e-8 else np.zeros_like(vals)
        scores[top_pool] = scores[top_pool] + float(self.cfg["stability_model"].get("weight", 0.0)) * z
        threshold = self.cfg["stability_model"].get("max_mean_degradation")
        fail_penalty = float(self.cfg["stability_model"].get("fail_penalty", 0.0))
        for rank, idx in enumerate(top_pool):
            idx = int(idx)
            stability_scores[idx] = vals[rank]
            stability_records[idx] = scored[rank]
            if threshold is not None and scored[rank]["rnadegformer_mean_degradation"] > float(threshold):
                scores[idx] -= fail_penalty
        return stability_scores, stability_records, float("nan") if threshold is None else float(threshold)

    def summarize(self, pred_pair: tuple[np.ndarray, np.ndarray]) -> dict:
        normal, stress = pred_pair
        uorf_names = self.design.uorf_names
        u_norm = np.array([normal[self.design.site_indices[name]] for name in uorf_names], dtype=float)
        u_stress = np.array([stress[self.design.site_indices[name]] for name in uorf_names], dtype=float)
        luc_idx = self.design.site_indices[self.design.luc_name]
        n_luc, s_luc = float(normal[luc_idx]), float(stress[luc_idx])
        diff_sites = self.constraints.get("uorf_difference_sites", [])
        diff_sum = sum(float(normal[self.design.site_indices[name]] - stress[self.design.site_indices[name]]) for name in diff_sites)
        return {
            "normal_uorf_mean": float(u_norm.mean()) if len(u_norm) else 0.0,
            "normal_luciferase": n_luc,
            "stress_uorf_mean": float(u_stress.mean()) if len(u_stress) else 0.0,
            "stress_luciferase": s_luc,
            "luciferase_induction": float(s_luc - n_luc),
            "uorf_difference_sum": float(diff_sum),
            "normal_internal_max": float(normal[self.internal_mask].max()),
            "stress_internal_max": float(stress[self.internal_mask].max()),
            "normal_max": float(normal.max()),
            "normal_max_pos": int(normal.argmax() + 1),
            "stress_max": float(stress.max()),
            "stress_max_pos": int(stress.argmax() + 1),
            **{f"normal_{self.safe_key(site['name'])}": float(normal[self.design.site_indices[site["name"]]]) for site in self.design.sites},
            **{f"stress_{self.safe_key(site['name'])}": float(stress[self.design.site_indices[site["name"]]]) for site in self.design.sites},
            **{
                f"stress_minus_normal_{self.safe_key(site['name'])}": float(
                    stress[self.design.site_indices[site["name"]]] - normal[self.design.site_indices[site["name"]]]
                )
                for site in self.design.sites
            },
            **{
                f"normal_minus_stress_{self.safe_key(site['name'])}": float(
                    normal[self.design.site_indices[site["name"]]] - stress[self.design.site_indices[site["name"]]]
                )
                for site in self.design.sites
            },
        }

    def objective(self, s: dict) -> float:
        w = self.weights
        internal = max(s["normal_internal_max"], s["stress_internal_max"])
        score = (
            w["desired_pair_min"] * min(s["normal_uorf_mean"], s["stress_luciferase"])
            + w["normal_uorf_mean"] * s["normal_uorf_mean"]
            + w["stress_luciferase"] * s["stress_luciferase"]
            + w["luciferase_induction"] * s["luciferase_induction"]
            + w["uorf_difference_sum"] * s["uorf_difference_sum"]
            - w["normal_luciferase_penalty"] * s["normal_luciferase"]
            - w["stress_uorf_mean_penalty"] * s["stress_uorf_mean"]
            - w["internal_tis_penalty"] * internal
        )
        for item in self.cfg.get("site_objectives", []):
            value = self.metric_value(s, item)
            score += float(item.get("weight", 1.0)) * value if item.get("direction", "high") == "high" else -float(item.get("weight", 1.0)) * value
        for item in self.cfg.get("metric_target_objectives", []):
            value = self.metric_value(s, item)
            weight = float(item.get("weight", 1.0))
            if "min" in item:
                score -= weight * max(0.0, float(item["min"]) - value)
            if "max" in item:
                score -= weight * max(0.0, value - float(item["max"]))
        return score

    def phenotype_success(self, s: dict) -> bool:
        c = self.constraints
        fixed_success = (
            s["normal_uorf_mean"] >= c["normal_uorf_mean_min"]
            and s["normal_luciferase"] <= c["normal_luciferase_max"]
            and s["stress_luciferase"] >= c["stress_luciferase_min"]
            and s["stress_uorf_mean"] <= c["stress_uorf_mean_max"]
            and s["luciferase_induction"] >= c["luciferase_induction_min"]
            and s["uorf_difference_sum"] >= c["uorf_difference_sum_min"]
        )
        if not fixed_success:
            return False
        for item in self.cfg.get("site_constraints", []):
            value = self.metric_value(s, item)
            if "min" in item and value < float(item["min"]):
                return False
            if "max" in item and value > float(item["max"]):
                return False
        return True

    def mutate(self, seq: str, rng=np.random) -> str:
        chars = list(seq)
        for pos in self.design.mutable_positions:
            pos = int(pos)
            rate = self.ga["local_mutation_rate"] if self.local_mask[pos] else self.ga["mutation_rate"]
            if rng.random() < rate:
                chars[pos] = str(rng.choice([base for base in BASES if base != chars[pos]]))
        return self.design.clean_sequence("".join(chars))

    def crossover(self, a: str, b: str, rng=np.random) -> str:
        child = list(a)
        for pos in self.design.mutable_positions:
            if rng.random() < 0.5:
                child[int(pos)] = b[int(pos)]
        return self.design.clean_sequence("".join(child))

    def tournament(self, population: list[str], scores: np.ndarray, rng=np.random) -> str:
        idxs = rng.choice(len(population), size=int(self.ga["tournament_k"]), replace=False)
        return population[int(idxs[np.argmax(scores[idxs])])]

    def encode_for_bo(self, sequences: list[str]) -> np.ndarray:
        positions = self.design.mutable_positions.astype(int)
        base_to_int = np.full(91, -1, dtype=np.int8)
        for idx, base in enumerate("ACGT"):
            base_to_int[ord(base)] = idx
        x = np.zeros((len(sequences), len(positions) * 4), dtype=np.float32)
        for row, seq in enumerate(sequences):
            codes = base_to_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)[positions]]
            cols = np.arange(len(positions)) * 4 + codes
            x[row, cols] = 1.0
        return x

    def update_bo_observations(self, population: list[str], scores: np.ndarray) -> None:
        if self.bo.get("enabled", False):
            self.bo_sequences.extend(population)
            self.bo_scores.extend(float(v) for v in scores)

    def bo_candidate_pool(self, elites: list[str]) -> list[str]:
        pool_size = int(self.bo.get("candidate_pool_size", 4096))
        seed_elites = elites[: max(1, min(len(elites), int(self.bo.get("seed_elite_count", 32))))]
        candidates: list[str] = []
        seen: set[str] = set()
        while len(candidates) < pool_size:
            r = np.random.random()
            if r < float(self.bo.get("mutated_elite_fraction", 0.55)):
                seq = self.mutate(seed_elites[int(np.random.randint(len(seed_elites)))])
            elif r < float(self.bo.get("mutated_elite_fraction", 0.55)) + float(self.bo.get("crossover_fraction", 0.30)):
                a = seed_elites[int(np.random.randint(len(seed_elites)))]
                b = seed_elites[int(np.random.randint(len(seed_elites)))]
                seq = self.mutate(self.crossover(a, b))
            else:
                seq = self.design.random_sequence()
            if seq not in seen:
                seen.add(seq)
                candidates.append(seq)
        return candidates

    def propose_bo_sequences(self, gen: int, elites: list[str], n: int) -> list[str]:
        if n <= 0 or not self.bo.get("enabled", False) or ExtraTreesRegressor is None:
            return []
        if gen < int(self.bo.get("start_generation", 20)) or gen % int(self.bo.get("interval", 5)) != 0:
            return []
        if len(self.bo_scores) < int(self.bo.get("min_observations", 2048)):
            return []
        max_samples = min(len(self.bo_scores), int(self.bo.get("train_max_samples", 12000)))
        if max_samples < len(self.bo_scores):
            probs = np.asarray(self.bo_scores, dtype=np.float64)
            probs = probs - probs.min() + 1e-6
            probs = probs / probs.sum() if probs.sum() > 0 else None
            train_idx = np.random.choice(len(self.bo_scores), size=max_samples, replace=False, p=probs)
            train_sequences = [self.bo_sequences[int(i)] for i in train_idx]
            train_scores = np.asarray([self.bo_scores[int(i)] for i in train_idx], dtype=np.float32)
        else:
            train_sequences = self.bo_sequences
            train_scores = np.asarray(self.bo_scores, dtype=np.float32)

        model = ExtraTreesRegressor(
            n_estimators=int(self.bo.get("n_estimators", 64)),
            max_features=self.bo.get("max_features", "sqrt"),
            min_samples_leaf=int(self.bo.get("min_samples_leaf", 2)),
            bootstrap=bool(self.bo.get("bootstrap", True)),
            random_state=int(self.bo.get("random_state", 0)) + gen,
            n_jobs=int(self.bo.get("n_jobs", -1)),
        )
        model.fit(self.encode_for_bo(train_sequences), train_scores)
        pool = self.bo_candidate_pool(elites)
        x_pool = self.encode_for_bo(pool)
        tree_predictions = np.vstack([tree.predict(x_pool) for tree in model.estimators_])
        acquisition = tree_predictions.mean(axis=0) + float(self.bo.get("exploration_beta", 0.75)) * tree_predictions.std(axis=0)
        proposals: list[str] = []
        seen = set(elites)
        for idx in np.argsort(acquisition)[::-1]:
            seq = pool[int(idx)]
            if seq not in seen:
                proposals.append(seq)
                seen.add(seq)
            if len(proposals) >= n:
                break
        return proposals

    def run(self) -> dict:
        threshold = self.estimate_evo2_threshold()
        pop_size = int(self.ga["population_size"])
        initial_count = int(self.evo_cfg.get("initial_seed_count", 0))
        evo_raw = self.evo2_generate_raw(initial_count) if initial_count else []
        population = [self.design.overlay_fixed(seq) for seq in evo_raw]
        population += [self.design.random_sequence() for _ in range(pop_size - len(population))]
        history, records, profile_records, top_profile_records = [], [], [], []
        best, best_constrained = None, None
        last_bo_proposals = 0
        stability_threshold = float("nan")
        for gen in range(int(self.ga["generations"]) + 1):
            preds = self.exp038.predict_batch(population, int(self.ga["batch_size"]))
            summaries = [self.summarize(pred) for pred in preds]
            exp_scores = np.array([self.objective(s) for s in summaries])
            scores = exp_scores.copy()
            evo_scores = np.full(pop_size, np.nan)
            stability_scores, stability_records, stability_threshold = self.add_stability_scores(population, exp_scores, scores)
            if self.constraint_mode == "evo2":
                top_pool = np.argsort(exp_scores)[::-1][: int(self.evo_cfg["topk_per_generation"])]
                top_evo = self.evo2_score([population[int(i)] for i in top_pool])
                evo_scores[top_pool] = top_evo
                evo_z = (top_evo - top_evo.mean()) / top_evo.std(ddof=1) if len(top_evo) > 1 and top_evo.std(ddof=1) > 1e-8 else np.zeros_like(top_evo)
                scores[top_pool] = scores[top_pool] + float(self.evo_cfg.get("weight", 0.0)) * evo_z
                scores[top_pool[top_evo < threshold]] -= float(self.evo_cfg.get("fail_penalty", 0.0))
            self.update_bo_observations(population, scores)
            order = np.argsort(scores)[::-1]
            best_i = int(order[0])
            cand = (population[best_i], float(scores[best_i]), preds[best_i], summaries[best_i], None if np.isnan(evo_scores[best_i]) else float(evo_scores[best_i]), stability_records.get(best_i))
            if best is None or cand[1] > best[1]:
                best = cand
            for idx in order:
                idx = int(idx)
                ev = None if np.isnan(evo_scores[idx]) else float(evo_scores[idx])
                stability_record = stability_records.get(idx)
                stability_ok = (
                    not self.stability.enabled
                    or stability_record is not None
                    and (
                        self.cfg["stability_model"].get("max_mean_degradation") is None
                        or stability_record["rnadegformer_mean_degradation"] <= float(self.cfg["stability_model"]["max_mean_degradation"])
                    )
                )
                evo_ok = self.constraint_mode != "evo2" or (ev is not None and ev >= threshold)
                if self.phenotype_success(summaries[idx]) and evo_ok and stability_ok:
                    cand2 = (population[idx], float(scores[idx]), preds[idx], summaries[idx], ev, stability_record)
                    if best_constrained is None or cand2[1] > best_constrained[1]:
                        best_constrained = cand2
                    break
            best_stability = stability_records.get(best_i, {})
            row = {
                "generation": gen,
                "best_score": float(scores[best_i]),
                "best_exp038_score": float(exp_scores[best_i]),
                "best_evo2_mean_logprob": None if np.isnan(evo_scores[best_i]) else float(evo_scores[best_i]),
                "best_evo2_pass": bool((not np.isnan(evo_scores[best_i])) and evo_scores[best_i] >= threshold),
                "best_rnadegformer_stability_score": None if np.isnan(stability_scores[best_i]) else float(stability_scores[best_i]),
                "best_rnadegformer_mean_degradation": best_stability.get("rnadegformer_mean_degradation"),
                "global_best_constrained_score": np.nan if best_constrained is None else best_constrained[1],
                "bo_proposals_from_previous_generation": int(last_bo_proposals),
                "phenotype_success": bool(self.phenotype_success(summaries[best_i])),
                **summaries[best_i],
                **best_stability,
            }
            history.append(row)
            bn, bs = preds[best_i]
            for p in range(self.design.length):
                profile_records.append({"generation": gen, "position_1based": p + 1, "prediction_normal": float(bn[p]), "prediction_stress": float(bs[p])})
            for rank, idx in enumerate(order[: int(self.cfg.get("gif", {}).get("right_profile_count", 5))]):
                normal_profile, stress_profile = preds[int(idx)]
                for p in range(self.design.length):
                    top_profile_records.append({"generation": gen, "rank": rank, "position_1based": p + 1, "prediction_normal": float(normal_profile[p]), "prediction_stress": float(stress_profile[p])})
            for rank, idx in enumerate(order):
                idx = int(idx)
                ev = None if np.isnan(evo_scores[idx]) else float(evo_scores[idx])
                rec_stability = stability_records.get(idx)
                records.append({
                    "generation": gen,
                    "rank": rank,
                    "sequence": population[idx],
                    "score": float(scores[idx]),
                    "exp038_score": float(exp_scores[idx]),
                    "evo2_mean_logprob": ev,
                    "evo2_pass": bool(ev is not None and ev >= threshold),
                    "rnadegformer_stability_score": None if np.isnan(stability_scores[idx]) else float(stability_scores[idx]),
                    "phenotype_success": bool(self.phenotype_success(summaries[idx])),
                    "constrained_success": bool(self.phenotype_success(summaries[idx])),
                    **summaries[idx],
                    **(rec_stability or {}),
                })
            print(row, flush=True)
            if gen == int(self.ga["generations"]):
                break
            elites = [population[int(i)] for i in order[: int(self.ga["elite_count"])]]
            nxt = elites[:]
            random_immigrants = int(self.ga["random_immigrants"])
            bo_count = max(0, min(int(self.bo.get("proposals_per_generation", 0)) if self.bo.get("enabled", False) else 0, random_immigrants))
            while len(nxt) < pop_size - random_immigrants:
                nxt.append(self.mutate(self.crossover(self.tournament(population, scores), self.tournament(population, scores))))
            proposals = self.propose_bo_sequences(gen, elites, bo_count)
            last_bo_proposals = len(proposals)
            nxt.extend(proposals)
            while len(nxt) < pop_size:
                nxt.append(self.design.random_sequence())
            population = nxt[:pop_size]
        selected = best_constrained if best_constrained is not None else best
        seq, score, pred, summary, ev, stability_record = selected
        if ev is None and self.constraint_mode == "evo2":
            ev = float(self.evo2_score([seq])[0])
        if self.stability.enabled and stability_record is None:
            stability_record = self.stability.score_one(seq)
        return {
            "threshold": threshold,
            "stability_threshold": stability_threshold,
            "selected_source": "best_constrained_success" if best_constrained is not None else "best_overall_no_constrained_success",
            "selected": (seq, score, pred, summary, ev, stability_record),
            "history": pd.DataFrame(history),
            "records": pd.DataFrame(records),
            "profiles": pd.DataFrame(profile_records),
            "top_profiles": pd.DataFrame(top_profile_records),
        }

    def write_outputs(self, result: dict) -> dict:
        seq, score, pred, summary, ev, stability_record = result["selected"]
        normal, stress = pred
        pd.DataFrame({
            "position_1based": np.arange(1, self.design.length + 1),
            "base": list(seq),
            "label": self.design.labels,
            "fixed": self.design.fixed,
            "prediction_normal": normal,
            "prediction_stress": stress,
        }).to_csv(self.out / "best_sequence_predictions.csv", index=False)
        result["history"].to_csv(self.out / "ga_history.csv", index=False)
        result["records"].to_csv(self.out / "ga_population_scores.csv", index=False)
        result["profiles"].to_csv(self.out / "ga_best_profiles.csv", index=False)
        result["top_profiles"].to_csv(self.out / "ga_top_profiles.csv", index=False)
        summary_json = {
            "config": self.cfg,
            "evo2_threshold": float(result["threshold"]),
            "naturalness_constraint": self.constraint_mode,
            "selected_source": result["selected_source"],
            "score": float(score),
            "evo2_mean_logprob": None if ev is None else float(ev),
            "evo2_pass": None if ev is None else bool(ev >= result["threshold"]),
            "stability": stability_record or {},
            "phenotype_success": bool(self.phenotype_success(summary)),
            "constrained_success": bool(self.phenotype_success(summary)),
            "target_predictions": summary,
            "sequence": seq,
        }
        (self.out / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
        with (self.out / "best_sequence.fa").open("w", encoding="utf-8") as handle:
            handle.write(">translation_switch_optimized\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + "\n")
        if self.cfg.get("gif", {}).get("enabled", True):
            self.write_gif()
        return summary_json

    def write_gif(self) -> None:
        gif_cfg = self.cfg.get("gif", {})
        pop = pd.read_csv(self.out / "ga_population_scores.csv")
        top_prof = pd.read_csv(self.out / "ga_top_profiles.csv")
        gif_df = pop[pop["rank"] < int(gif_cfg.get("top_rank", 128))].copy()
        generations = sorted(gif_df["generation"].unique())[:: int(gif_cfg.get("frame_step", 1))]
        x_col, y_col = "normal_uorf_mean", "stress_luciferase"
        x_max = max(float(gif_df[x_col].max()), 1e-3) * 1.08
        y_max = max(float(gif_df[y_col].max()), 1e-3) * 1.08
        profile_y_max = max(float(top_prof["prediction_normal"].max()), float(top_prof["prediction_stress"].max()), 1e-3) * 1.12
        profile_count = max(1, int(gif_cfg.get("right_profile_count", 5)))
        fig = plt.figure(figsize=(13.5, max(5.2, 1.45 * profile_count)))
        grid = fig.add_gridspec(profile_count, 2, width_ratios=[1.05, 1.45], hspace=0.34, wspace=0.26)
        left_ax = fig.add_subplot(grid[:, 0])
        profile_axes = [fig.add_subplot(grid[i, 1]) for i in range(profile_count)]

        def update(frame):
            gen = generations[frame]
            current = gif_df[gif_df["generation"] == gen]
            profiles = top_prof[top_prof["generation"] == gen]
            ax = left_ax
            ax.clear()
            ax.scatter(
                current[x_col],
                current[y_col],
                color="tab:blue",
                s=float(gif_cfg.get("point_size", 18)),
                alpha=float(gif_cfg.get("point_alpha", 0.72)),
                edgecolors="none",
                label=f"top {int(gif_cfg.get('top_rank', 128))} candidates",
            )
            success = current[current["constrained_success"]]
            if len(success):
                ax.scatter(
                    success[x_col],
                    success[y_col],
                    s=52,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    label="all constraints met",
                )
            generation_best = current.loc[current["rank"].idxmin()]
            ax.scatter(
                generation_best[x_col],
                generation_best[y_col],
                color="crimson",
                s=72,
                marker="*",
                zorder=3,
                label="generation best",
            )
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("normal uORF mean TIS")
            ax.set_ylabel("stress luciferase TIS")
            ax.set_title(f"Top candidates, generation {gen}")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.25)

            for rank, ax in enumerate(profile_axes):
                ax.clear()
                profile = profiles[profiles["rank"] == rank]
                if len(profile):
                    ax.plot(profile["position_1based"], profile["prediction_normal"], lw=1.2, label="normal")
                    ax.plot(profile["position_1based"], profile["prediction_stress"], lw=1.2, label="stress")
                for site in self.design.sites:
                    color = "tab:orange" if site.get("group") == "luciferase" else "tab:blue"
                    ax.axvline(site["position"], color=color, ls="--", lw=0.9, alpha=0.60)
                ax.set_xlim(1, self.design.length)
                ax.set_ylim(0, profile_y_max)
                ax.set_ylabel(f"rank {rank + 1}")
                ax.grid(alpha=0.18)
                if rank == 0:
                    ax.set_title("Top candidate profiles in this generation")
                    ax.legend(loc="upper right", fontsize=8)
                if rank == profile_count - 1:
                    ax.set_xlabel("Position")
                else:
                    ax.tick_params(labelbottom=False)
            fig.tight_layout()
            return []

        anim = FuncAnimation(fig, update, frames=len(generations), interval=int(gif_cfg.get("interval_ms", 120)), blit=False)
        anim.save(self.out / "top_candidates_with_profile.gif", writer=PillowWriter(fps=int(gif_cfg.get("fps", 8))))
        update(len(generations) - 1)
        fig.savefig(self.out / "top_candidates_with_profile_final.png", dpi=170)
        plt.close(fig)
