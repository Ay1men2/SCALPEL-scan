"""
Statistical significance utilities for SCALPEL experiments.

Provides bootstrap confidence intervals, majority voting across runs,
and aggregate statistics for multi-run experiments.

Usage:
    python -m scalpel.stats --runs dir1 dir2 dir3
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def bootstrap_ci(values: list, n_bootstrap: int = 10000,
                 confidence: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sequence of numeric observations to bootstrap over.
        n_bootstrap: Number of bootstrap resamples to draw. Defaults to
            10 000.
        confidence: Confidence level for the interval, expressed as a
            fraction in (0, 1). Defaults to 0.95 (95 % CI).
        seed: Random seed for the bootstrap RNG. Defaults to 42 for
            reproducibility.

    Returns:
        Tuple of (mean, ci_low, ci_high) where *mean* is the sample mean
        and *ci_low* / *ci_high* are the lower and upper percentile bounds
        of the bootstrap distribution.
    """
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(values.mean())

    return (mean, ci_low, ci_high)


def _load_judge_results(run_dir: Path) -> list[dict]:
    """Load judge_results.jsonl from a run directory.

    Falls back to reading classifications from ``summary.json`` when the
    JSONL file is not present.

    Args:
        run_dir: Path to a strike-run directory that contains either
            ``judge_results.jsonl`` or ``summary.json``.

    Returns:
        List of dicts, each containing at least ``prompt_idx`` and the
        judge label (keyed as ``judge_label`` or ``label``).

    Raises:
        FileNotFoundError: If neither ``judge_results.jsonl`` nor
            ``summary.json`` exists in *run_dir*.
    """
    path = run_dir / "judge_results.jsonl"
    if not path.exists():
        # Fall back to summary.json classifications
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            return [
                {"prompt_idx": c["prompt_idx"], "judge_label": c["label"]}
                for c in summary.get("classifications", [])
            ]
        raise FileNotFoundError(f"No judge results or summary at {run_dir}")

    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def majority_vote(run_dirs: list[Path]) -> dict:
    """Compute majority-vote ASR across multiple runs.

    For each prompt, takes the majority label across runs,
    then computes ASR from the majority-voted labels.

    Args:
        run_dirs: List of Path objects pointing to strike-run directories,
            each containing judge results loadable by
            ``_load_judge_results``.

    Returns:
        Dict with keys ``asr`` (float), ``total`` (int), ``bypass`` (int),
        ``num_runs`` (int), and ``per_prompt`` (list of dicts each holding
        ``prompt_idx``, ``votes``, and ``majority_label``).
    """
    # Collect labels per prompt across runs
    prompt_votes: dict[int, list[str]] = {}
    for rd in run_dirs:
        results = _load_judge_results(rd)
        for r in results:
            idx = r["prompt_idx"]
            label = r.get("judge_label", r.get("label", "unknown"))
            prompt_votes.setdefault(idx, []).append(label)

    # Majority vote
    per_prompt = []
    bypass_count = 0
    for idx in sorted(prompt_votes.keys()):
        votes = prompt_votes[idx]
        counts = Counter(votes)
        majority = counts.most_common(1)[0][0]
        per_prompt.append({
            "prompt_idx": idx,
            "votes": dict(counts),
            "majority_label": majority,
        })
        if majority == "bypass":
            bypass_count += 1

    total = len(per_prompt)
    return {
        "asr": round(bypass_count / total, 4) if total > 0 else 0,
        "total": total,
        "bypass": bypass_count,
        "num_runs": len(run_dirs),
        "per_prompt": per_prompt,
    }


def aggregate_runs(run_dirs: list[Path]) -> dict:
    """Compute aggregate statistics across multiple runs.

    Pools per-prompt binary outcomes (bypass=1, else=0) from all runs and
    bootstraps over these individual outcomes for a statistically meaningful
    confidence interval.  Also reports per-run ASRs for transparency.

    Args:
        run_dirs: List of Path objects pointing to strike-run directories,
            each containing judge results loadable by
            ``_load_judge_results``.

    Returns:
        Dict with keys ``num_runs`` (int), ``per_run_asr`` (list of float),
        ``mean_asr`` (float), and ``ci_95`` (tuple of two floats giving
        the 95 % bootstrap confidence interval bounds).
    """
    per_run_asrs = []
    all_outcomes = []  # pooled per-prompt binary outcomes across all runs
    for rd in run_dirs:
        results = _load_judge_results(rd)
        total = len(results)
        bypass = 0
        for r in results:
            is_bypass = r.get("judge_label", r.get("label")) == "bypass"
            all_outcomes.append(1.0 if is_bypass else 0.0)
            if is_bypass:
                bypass += 1
        asr = bypass / total if total > 0 else 0
        per_run_asrs.append(asr)

    # Bootstrap over pooled per-prompt outcomes for meaningful CI
    mean, ci_low, ci_high = bootstrap_ci(all_outcomes)

    return {
        "num_runs": len(run_dirs),
        "per_run_asr": [round(a, 4) for a in per_run_asrs],
        "mean_asr": round(mean, 4),
        "ci_95": (round(ci_low, 4), round(ci_high, 4)),
    }


def energy_gini(scan_data: dict) -> float:
    """Gini coefficient of per-layer energy. 0=uniform, 1=single-layer.

    Parameters
    ----------
    scan_data : dict
        Loaded energy_map.json with "heads" list and "num_layers".

    Returns
    -------
    float
        Gini coefficient in [0, 1]. A value of 0 indicates perfectly
        uniform energy across layers; 1 indicates all energy is
        concentrated in a single layer.
    """
    num_layers = scan_data["num_layers"]
    layer_energy = np.zeros(num_layers)
    for h in scan_data["heads"]:
        layer_energy[h["layer"]] += h.get("energy", h.get("selectivity", 0))
    s = np.sort(layer_energy)
    n = len(s)
    total = np.sum(s)
    if total == 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s)) / (n * total) - (n + 1) / n)


def main():
    """CLI entry point for statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Statistical significance analysis for SCALPEL runs"
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to strike run directories")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    print(f"[STATS] Analyzing {len(run_dirs)} runs")

    # Aggregate
    agg = aggregate_runs(run_dirs)
    print(f"\n  Per-run ASR: {agg['per_run_asr']}")
    print(f"  Mean ASR:    {agg['mean_asr']:.2%}")
    print(f"  95% CI:      [{agg['ci_95'][0]:.2%}, {agg['ci_95'][1]:.2%}]")

    # Majority vote
    mv = majority_vote(run_dirs)
    print(f"\n  Majority-vote ASR: {mv['asr']:.2%} ({mv['bypass']}/{mv['total']})")

    return {"aggregate": agg, "majority_vote": mv}


if __name__ == "__main__":
    main()
