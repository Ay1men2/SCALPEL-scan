"""
Phase 2: PLAN — Generate attack plan from scan results.

Six modes:
  full_uniform:          All heads with selectivity > threshold, same α for all.
  targeted_uniform:      Only Z-score outlier heads, same α.
  targeted_adaptive:     Only Z-score outlier heads, per-head α via power-law.
  full_adaptive:         All heads, theory-guided α*(l,h) = 1 + 1/Sel². No hyperparams.
  full_adaptive_clipped: Same as full_adaptive but clamped to [alpha_min, alpha_max].
  full_depth_adaptive:   All heads, α scaled by a depth-dependent decay schedule.
"""

import json
import math
import time
from pathlib import Path

import numpy as np


# ─── Plan Generation ─────────────────────────────────────────────────────────

def generate_plan(
    energy_map: dict,
    mode: str = "full_uniform",
    alpha: float = 2.0,
    z_threshold: float = 2.0,
    selectivity_floor: float = 1.0,  # minimum selectivity to include (full_uniform)
    gamma: float = 0.5,              # power-law compression (adaptive mode)
    cap: float = 3.0,                # max alpha = cap × alpha_base
    alpha_min: float = 1.0,          # min alpha for full_adaptive_clipped
    alpha_max: float = 4.0,          # max alpha for full_adaptive_clipped
    decay: str = "linear",           # decay schedule for full_depth_adaptive
) -> dict:
    """Generate attack plan from energy map.

    Selects attention heads to target and assigns per-head erasure strengths
    (alpha values) according to the chosen ``mode``.  Returns a plan dict
    containing the list of targets and summary metadata suitable for the
    strike phase.

    Args:
        energy_map (dict): Output of the scan phase.  Must contain keys
            ``"heads"`` (list of head dicts with ``"layer"``, ``"head"``,
            ``"selectivity"``, ``"z_score"``), ``"num_layers"``, and
            ``"num_kv_heads"``.
        mode (str): Selection/scaling strategy.  One of ``"full_uniform"``,
            ``"targeted_uniform"``, ``"targeted_adaptive"``,
            ``"full_adaptive"``, ``"full_adaptive_clipped"``, or
            ``"full_depth_adaptive"``.
        alpha (float): Base erasure strength applied uniformly or used as
            a multiplier in adaptive modes.
        z_threshold (float): Minimum Z-score for a head to be considered
            an outlier in targeted modes.
        selectivity_floor (float): Minimum selectivity required to include
            a head in full (non-targeted) modes.
        gamma (float): Power-law compression exponent used by
            ``targeted_adaptive`` mode.  Values < 1 compress the range.
        cap (float): Maximum ratio multiplier in ``targeted_adaptive``
            (head alpha capped at ``alpha * cap``).
        alpha_min (float): Lower clamp for ``full_adaptive_clipped`` mode.
        alpha_max (float): Upper clamp for ``full_adaptive_clipped`` mode.
        decay (str): Depth-decay schedule for ``full_depth_adaptive`` mode.
            One of ``"linear"``, ``"cosine"``, ``"halflife"``, or
            ``"late_focused"``.

    Returns:
        dict: A plan dictionary with keys ``"model"``, ``"mode"``,
        ``"alpha_base"``, ``"params"``, ``"timestamp"``, ``"summary"``,
        and ``"targets"``.
    """
    heads = energy_map["heads"]
    num_layers = energy_map["num_layers"]
    num_kv_heads = energy_map["num_kv_heads"]
    total_heads = num_layers * num_kv_heads
    
    targets = []
    
    if mode == "full_uniform":
        # Full uniform: every head above the selectivity floor gets the same
        # constant alpha.  Simplest mode -- useful as a baseline.
        for h in heads:
            if h["selectivity"] > selectivity_floor:
                targets.append({
                    "layer": h["layer"], "head": h["head"],
                    "alpha": alpha,
                    "selectivity": h["selectivity"],
                    "z_score": h["z_score"],
                })
    
    elif mode == "targeted_uniform":
        # Targeted uniform: restrict to Z-score outlier heads only, then
        # apply a single constant alpha.  Reduces collateral damage.
        for h in heads:
            if h["z_score"] > z_threshold:
                targets.append({
                    "layer": h["layer"], "head": h["head"],
                    "alpha": alpha,
                    "selectivity": h["selectivity"],
                    "z_score": h["z_score"],
                })
    
    elif mode == "targeted_adaptive":
        # Targeted adaptive: Z-score outlier heads with per-head alpha
        # scaled via a power-law relative to the median outlier selectivity.
        outlier_heads = [h for h in heads if h["z_score"] > z_threshold]
        if not outlier_heads:
            print(f"[PLAN] WARNING: No heads with Z > {z_threshold}. Try lower threshold.")
            return _empty_plan(energy_map, mode, alpha)
        
        # Anchor = median selectivity of outlier heads
        outlier_sels = [h["selectivity"] for h in outlier_heads]
        anchor = float(np.median(outlier_sels))
        
        for h in outlier_heads:
            ratio = (h["selectivity"] / anchor) ** gamma
            h_alpha = min(alpha * ratio, alpha * cap)
            targets.append({
                "layer": h["layer"], "head": h["head"],
                "alpha": round(h_alpha, 4),
                "selectivity": h["selectivity"],
                "z_score": h["z_score"],
            })
    elif mode == "full_adaptive":
        # Full adaptive: pure theory-guided alpha with no tunable
        # hyperparameters.  Each head's alpha is derived directly from the
        # noise compensation model: alpha(l,h) = 1 + 1/Sel(l,h)^2.
        from .adaptive_alpha import alpha_from_selectivity
        for h in heads:
            if h["selectivity"] > selectivity_floor:
                targets.append({
                    "layer": h["layer"], "head": h["head"],
                    "alpha": round(alpha_from_selectivity(h["selectivity"]), 4),
                    "selectivity": h["selectivity"],
                    "z_score": h["z_score"],
                })

    elif mode == "full_adaptive_clipped":
        # Full adaptive clipped: same theory-guided formula as full_adaptive,
        # but the resulting alpha is clamped to [alpha_min, alpha_max] to
        # prevent extreme values from low-selectivity heads.
        from .adaptive_alpha import alpha_from_selectivity
        for h in heads:
            if h["selectivity"] > selectivity_floor:
                a = alpha_from_selectivity(h["selectivity"])
                a = max(alpha_min, min(alpha_max, a))
                targets.append({
                    "layer": h["layer"], "head": h["head"],
                    "alpha": round(a, 4),
                    "selectivity": h["selectivity"],
                    "z_score": h["z_score"],
                })

    elif mode == "full_depth_adaptive":
        # Full depth adaptive: alpha is multiplied by a depth-dependent
        # scale factor that decreases with layer index.  This prevents
        # cascading perturbation amplification in deep models (e.g. 70B).
        L = num_layers
        for h in heads:
            if h["selectivity"] > selectivity_floor:
                l = h["layer"]
                if decay == "linear":
                    # Linear decay: scale drops from 1.0 at layer 0 to 0.0
                    # at the final layer.
                    scale = 1.0 - l / L
                elif decay == "cosine":
                    # Cosine decay: smooth S-curve from 1.0 to 0.0 across
                    # the full depth, slower change at the extremes.
                    scale = 0.5 * (1.0 + math.cos(math.pi * l / L))
                elif decay == "halflife":
                    # Halflife decay: front half of layers is unmodified
                    # (scale=1.0 for l < L/2), then linearly ramps down
                    # to 0.0 at the final layer.
                    scale = 1.0 if l < L / 2 else (L - l) / (L / 2)
                elif decay == "late_focused":
                    # Late focused: intentionally opposite of other decay
                    # schedules.  Early layers get a low constant scale
                    # (0.3) while the last 10% of layers receive full
                    # strength (1.0), concentrating the attack on the
                    # model's deepest representations.
                    scale = 0.3 if l < L * 0.9 else 1.0
                else:
                    raise ValueError(f"Unknown decay: {decay}. Use: linear, cosine, halflife, late_focused")
                targets.append({
                    "layer": h["layer"], "head": h["head"],
                    "alpha": round(alpha * scale, 4),
                    "selectivity": h["selectivity"],
                    "z_score": h["z_score"],
                })

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use: full_uniform, targeted_uniform, "
            f"targeted_adaptive, full_adaptive, full_adaptive_clipped, "
            f"full_depth_adaptive"
        )
    
    # Sort by layer, then head
    targets.sort(key=lambda t: (t["layer"], t["head"]))
    
    # Compute summary
    layers_covered = sorted(set(t["layer"] for t in targets))
    alphas = [t["alpha"] for t in targets]
    
    plan = {
        "model": energy_map["model"],
        "mode": mode,
        "alpha_base": alpha,
        "params": {
            "z_threshold": z_threshold,
            "selectivity_floor": selectivity_floor,
            "gamma": gamma,
            "cap": cap,
            "decay": decay,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_heads": total_heads,
            "targeted_heads": len(targets),
            "targeted_pct": round(len(targets) / total_heads * 100, 1),
            "layers_covered": len(layers_covered),
            "layers_total": num_layers,
            "alpha_min": round(min(alphas), 4) if alphas else 0,
            "alpha_max": round(max(alphas), 4) if alphas else 0,
            "alpha_mean": round(float(np.mean(alphas)), 4) if alphas else 0,
        },
        "targets": targets,
    }
    
    return plan


def _empty_plan(energy_map, mode, alpha):
    """Return a zero-target fallback plan when no heads pass the selection filter."""
    return {
        "model": energy_map["model"], "mode": mode, "alpha_base": alpha,
        "summary": {"total_heads": 0, "targeted_heads": 0}, "targets": [],
    }


# ─── Print Plan ──────────────────────────────────────────────────────────────

def print_plan(plan: dict):
    """Print a formatted, human-readable attack plan summary to stdout.

    Displays the model name, mode, base alpha, head/layer coverage
    statistics, alpha range, and a per-layer head count breakdown.

    Args:
        plan (dict): Plan dictionary as returned by ``generate_plan``.
    """
    s = plan["summary"]
    print()
    print("=" * 70)
    print(f"  ATTACK PLAN: {plan['model']}  |  mode={plan['mode']}  |  α_base={plan['alpha_base']}")
    print("=" * 70)
    print(f"  Targeted heads: {s['targeted_heads']}/{s['total_heads']} ({s['targeted_pct']}%)")
    print(f"  Layers covered: {s['layers_covered']}/{s['layers_total']}")
    print(f"  Alpha range:    [{s['alpha_min']}, {s['alpha_max']}]  mean={s['alpha_mean']}")
    
    # Per-layer breakdown
    targets = plan["targets"]
    if targets:
        layer_counts = {}
        for t in targets:
            l = t["layer"]
            layer_counts[l] = layer_counts.get(l, 0) + 1
        
        print(f"\n  Per-layer: ", end="")
        for l in sorted(layer_counts.keys()):
            print(f"L{l}({layer_counts[l]}h) ", end="")
        print()
    print()


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_plan(plan: dict, out_dir: Path) -> Path:
    """Serialize a plan dictionary to a JSON file on disk.

    The file is written to ``<out_dir>/<model_name>/plan_<mode>_<alpha>.json``.
    For ``full_depth_adaptive`` mode the decay schedule name is included in
    the filename to distinguish variants.

    Args:
        plan (dict): Plan dictionary as returned by ``generate_plan``.
        out_dir (Path): Root output directory.  A model-specific subdirectory
            is created automatically.

    Returns:
        Path: Absolute path to the written JSON file.
    """
    model_name = plan["model"]
    plan_dir = out_dir / model_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    
    decay = plan.get("params", {}).get("decay")
    if decay and plan["mode"] == "full_depth_adaptive":
        filename = f"plan_{plan['mode']}_{decay}_{plan['alpha_base']}.json"
    else:
        filename = f"plan_{plan['mode']}_{plan['alpha_base']}.json"
    path = plan_dir / filename
    
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)
    
    print(f"[PLAN] Saved to {path}")
    return path


def load_plan(plan_path: Path) -> dict:
    """Deserialize a plan dictionary from a JSON file on disk.

    Args:
        plan_path (Path): Path to a plan JSON file previously written by
            ``save_plan``.

    Returns:
        dict: The plan dictionary with keys ``"model"``, ``"mode"``,
        ``"alpha_base"``, ``"params"``, ``"timestamp"``, ``"summary"``,
        and ``"targets"``.
    """
    with open(plan_path) as f:
        return json.load(f)


def find_latest_plan(model_name: str, out_dir: Path) -> Path:
    """Find the most recently written plan file for a model.

    Sorts by file modification time (not alphabetically) so that the plan
    created by the immediately preceding run_plan() call is always selected,
    regardless of mode/alpha naming.
    """
    plan_dir = out_dir / model_name
    if not plan_dir.exists():
        raise FileNotFoundError(f"No plans found at {plan_dir}/")

    plans = sorted(plan_dir.glob("plan_*.json"), key=lambda p: p.stat().st_mtime)
    if not plans:
        raise FileNotFoundError(f"No plan files in {plan_dir}/")

    return plans[-1]


# ─── Main Entry ──────────────────────────────────────────────────────────────

def run_plan(model_name: str, scan_dir: Path, plan_dir: Path,
             mode: str = "full_uniform", alpha: float = 2.0,
             z_threshold: float = 2.0, gamma: float = 0.5, cap: float = 3.0,
             alpha_min: float = 1.0, alpha_max: float = 4.0,
             decay: str = "linear",
             scan_name: str = None) -> Path:
    """Load scan results and generate attack plan.

    Parameters
    ----------
    scan_name : str, optional
        Override scan directory name (e.g. "model-pairs8" for E7 ablation).
        Defaults to model_name.

    Returns the path to the saved plan file (pass to run_strike's plan_path).
    """
    _scan = scan_name or model_name

    # Load energy map (from scan_name directory)
    map_path = scan_dir / _scan / "energy_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"No energy map at {map_path}. Run: python -m scalpel scan --model {model_name}")

    with open(map_path) as f:
        energy_map = json.load(f)

    # Ensure plan uses the real model_name (not scan_tag)
    energy_map["model"] = model_name

    # Generate plan
    plan = generate_plan(energy_map, mode=mode, alpha=alpha,
                         z_threshold=z_threshold, gamma=gamma, cap=cap,
                         alpha_min=alpha_min, alpha_max=alpha_max,
                         decay=decay)

    # Print and save (plan is saved under model_name directory)
    print_plan(plan)
    plan_path = save_plan(plan, plan_dir)

    return plan_path
