"""
SCALPEL Plotting Module — Publication-quality figures.

Usage (no GPU required):
    python -m scalpel.plot --mode theory --out-dir figures/
    python -m scalpel.plot --mode all --scan-data outputs/scans/Model/energy_map.json --out-dir figures/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ─── Global Style ────────────────────────────────────────────────────────────

# Colorblind-friendly palette (Tol's muted)
COLORS = {
    "blue":   "#332288",
    "cyan":   "#88CCEE",
    "green":  "#44AA99",
    "yellow": "#DDCC77",
    "red":    "#CC6677",
    "purple": "#AA4499",
    "grey":   "#BBBBBB",
    "orange": "#EE8866",
}

COLOR_LIST = [COLORS["blue"], COLORS["red"], COLORS["green"],
              COLORS["yellow"], COLORS["cyan"], COLORS["purple"],
              COLORS["orange"], COLORS["grey"]]


def _setup_style():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, out_dir: Path, name: str):
    """Save figure as PDF and close.

    Args:
        fig: Matplotlib Figure instance to save.
        out_dir: Directory in which to write the PDF. Created
            recursively if it does not already exist.
        name: Stem filename (without extension) for the output PDF.

    Returns:
        Path to the saved PDF file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {path}")
    return path


# ─── 1. Selectivity Heatmap ─────────────────────────────────────────────────

def plot_selectivity_heatmap(energy_map: dict, out_dir: Path) -> Path:
    """Heatmap of per-head selectivity across layers.

    Parameters
    ----------
    energy_map : dict
        Loaded energy_map.json from scan phase.
    out_dir : Path
        Output directory for the figure.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    num_layers = energy_map["num_layers"]
    num_kv_heads = energy_map["num_kv_heads"]
    grid = np.zeros((num_layers, num_kv_heads))

    for h in energy_map["heads"]:
        grid[h["layer"], h["head"]] = h["selectivity"]

    fig, ax = plt.subplots(figsize=(max(6, num_kv_heads * 0.5), max(4, num_layers * 0.2)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("KV Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Selectivity Heatmap — {energy_map.get('model', 'Model')}")
    fig.colorbar(im, ax=ax, label="Selectivity")

    return _save(fig, out_dir, "selectivity_heatmap")


# ─── 2. Energy Distribution ─────────────────────────────────────────────────

def plot_energy_distribution(energy_map: dict, out_dir: Path) -> Path:
    """Histogram of energy values with cumulative distribution overlay.

    Returns:
        Path to the saved PDF file.
    """
    _setup_style()

    energies = np.array([h["energy"] for h in energy_map["heads"]])

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Histogram
    n_bins = min(50, max(10, len(energies) // 5))
    counts, bins, patches = ax1.hist(energies, bins=n_bins, color=COLORS["blue"],
                                     alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Energy")
    ax1.set_ylabel("Count", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    sorted_e = np.sort(energies)
    cumulative = np.arange(1, len(sorted_e) + 1) / len(sorted_e) * 100
    ax2.plot(sorted_e, cumulative, color=COLORS["red"], linewidth=1.5, label="Cumulative %")
    ax2.set_ylabel("Cumulative %", color=COLORS["red"])
    ax2.tick_params(axis="y", labelcolor=COLORS["red"])
    ax2.set_ylim(0, 105)

    ax1.set_title(f"Energy Distribution — {energy_map.get('model', 'Model')}")
    fig.tight_layout()

    return _save(fig, out_dir, "energy_distribution")


# ─── 3. Alpha Sensitivity ───────────────────────────────────────────────────

def plot_alpha_sensitivity(results: list, out_dir: Path) -> Path:
    """Dual-Y axis plot: ASR and Gibberish % vs alpha.

    Parameters
    ----------
    results : list of dict
        Each: {"alpha": float, "asr": float, "gibberish": float}

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    results = sorted(results, key=lambda r: r["alpha"])
    alphas = [r["alpha"] for r in results]
    asrs = [r["asr"] for r in results]
    gibbs = [r["gibberish"] for r in results]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(alphas, asrs, "o-", color=COLORS["red"], label="ASR", linewidth=2, markersize=6)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("ASR (%)", color=COLORS["red"])
    ax1.tick_params(axis="y", labelcolor=COLORS["red"])
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(alphas, gibbs, "s--", color=COLORS["blue"], label="Gibberish", linewidth=2, markersize=6)
    ax2.set_ylabel("Gibberish (%)", color=COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax2.set_ylim(0, 105)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(r"$\alpha$ Sensitivity")
    fig.tight_layout()

    return _save(fig, out_dir, "alpha_sensitivity")


# ─── 4. Layer Ablation ──────────────────────────────────────────────────────

def plot_layer_ablation(results: list, out_dir: Path) -> Path:
    """Horizontal bar chart for layer ablation study.

    Parameters
    ----------
    results : list of dict
        Each: {"name": str, "asr": float, "gibberish": float}

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    names = [r["name"] for r in results]
    asrs = [r["asr"] for r in results]
    gibbs = [r["gibberish"] for r in results]

    y = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
    ax.barh(y - height / 2, asrs, height, label="ASR", color=COLORS["red"], alpha=0.85)
    ax.barh(y + height / 2, gibbs, height, label="Gibberish", color=COLORS["blue"], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.set_title("Layer Ablation")
    ax.invert_yaxis()
    fig.tight_layout()

    return _save(fig, out_dir, "layer_ablation")


# ─── 5. HarmBench Categories ────────────────────────────────────────────────

def plot_harmbench_categories(categories: dict, out_dir: Path) -> Path:
    """Vertical bar chart of per-category ASR on HarmBench.

    Parameters
    ----------
    categories : dict
        {category_name: {"asr": float, "n": int}}

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    cats = sorted(categories.keys())
    asrs = [categories[c]["asr"] for c in cats]
    ns = [categories[c]["n"] for c in cats]

    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 0.8), 5))
    x = np.arange(len(cats))
    bars = ax.bar(x, asrs, color=COLORS["red"], alpha=0.85, edgecolor="white")

    # Annotate with sample count
    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 105)
    ax.set_title("HarmBench Per-Category ASR")
    fig.tight_layout()

    return _save(fig, out_dir, "harmbench_categories")


# ─── 6. Utility Comparison ──────────────────────────────────────────────────

def plot_utility_comparison(baseline: dict, scalpel: dict, out_dir: Path) -> Path:
    """Grouped bar chart comparing baseline vs SCALPEL utility metrics.

    Parameters
    ----------
    baseline : dict
        {"mmlu": float, "gsm8k": float, ...}  (accuracy percentages)
    scalpel : dict
        Same keys as baseline.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    metrics = sorted(set(baseline.keys()) & set(scalpel.keys()))
    base_vals = [baseline[m] for m in metrics]
    scal_vals = [scalpel[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(metrics) * 1.5), 4))
    ax.bar(x - width / 2, base_vals, width, label="Baseline", color=COLORS["blue"], alpha=0.85)
    ax.bar(x + width / 2, scal_vals, width, label="SCALPEL", color=COLORS["red"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("Utility Preservation")
    fig.tight_layout()

    return _save(fig, out_dir, "utility_comparison")


# ─── 7. Adaptive Alpha Plot ─────────────────────────────────────────────────

def plot_adaptive_alpha(energy_map: dict, out_dir: Path) -> Path:
    """1x2 subplot: (a) theoretical curve, (b) scatter of per-head alpha vs selectivity.

    Parameters
    ----------
    energy_map : dict
        Loaded energy_map.json from scan phase.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()
    from .adaptive_alpha import alpha_from_selectivity, theoretical_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Theoretical curve
    sel_arr, alpha_arr = theoretical_curve()
    ax1.plot(sel_arr, alpha_arr, color=COLORS["red"], linewidth=2)
    ax1.axhline(y=2.0, color=COLORS["grey"], linestyle="--", linewidth=1, label=r"$\alpha=2$ (global)")
    ax1.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=1, label=r"$\alpha=1$ (no erasure)")
    ax1.set_xlabel("Selectivity")
    ax1.set_ylabel(r"$\alpha^*$")
    ax1.set_title(r"(a) Theoretical: $\alpha^* = 1 + 1/\mathrm{Sel}^2$")
    ax1.set_ylim(0.8, 5)
    ax1.legend(fontsize=8)

    # (b) Scatter from actual scan data
    sels = [h["selectivity"] for h in energy_map["heads"]]
    alphas = [alpha_from_selectivity(s) for s in sels]
    layers = [h["layer"] for h in energy_map["heads"]]

    norm = Normalize(vmin=min(layers), vmax=max(layers))
    scatter = ax2.scatter(sels, alphas, c=layers, cmap="viridis", s=12, alpha=0.7, norm=norm)
    fig.colorbar(scatter, ax=ax2, label="Layer")

    # Overlay theory curve
    ax2.plot(sel_arr, alpha_arr, color=COLORS["red"], linewidth=1.5, alpha=0.5, zorder=0)

    ax2.set_xlabel("Selectivity")
    ax2.set_ylabel(r"$\alpha^*$")
    ax2.set_title(f"(b) Per-Head — {energy_map.get('model', 'Model')}")
    ax2.set_ylim(0.8, min(8, max(alphas) * 1.1))

    fig.tight_layout()

    return _save(fig, out_dir, "adaptive_alpha")


# ─── 8. Cross-Model ASR ─────────────────────────────────────────────────────

def plot_cross_model_asr(results: list, out_dir: Path) -> Path:
    """Horizontal bar chart of ASR and Gibberish across models.

    Parameters
    ----------
    results : list of dict
        Each: {"model": str, "asr": float, "gibberish": float}

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    models = [r["model"] for r in results]
    asrs = [r["asr"] for r in results]
    gibbs = [r["gibberish"] for r in results]

    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(7, max(3, len(models) * 0.6)))
    ax.barh(y - height / 2, asrs, height, label="ASR", color=COLORS["red"], alpha=0.85)
    ax.barh(y + height / 2, gibbs, height, label="Gibberish", color=COLORS["blue"], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.set_title("Cross-Model Comparison")
    ax.invert_yaxis()
    fig.tight_layout()

    return _save(fig, out_dir, "cross_model_asr")


# ─── 9. Distortion Ratio (Theoretical) ──────────────────────────────────────

def plot_distortion_ratio(out_dir: Path, f_min: float = 0.01, f_max: float = 1.0) -> Path:
    r"""Theoretical distortion ratio vs refusal fraction.

    Plots D(f) = alpha^2 * f + (1 - f) showing how total distortion
    varies with the fraction of KV entries carrying refusal signal.

    Parameters
    ----------
    f_min, f_max : float
        Range of refusal fraction f to plot.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    f = np.linspace(f_min, f_max, 200)
    alphas_to_plot = [1.5, 2.0, 3.0, 4.0]

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, a in enumerate(alphas_to_plot):
        D = a**2 * f + (1 - f)
        ax.plot(f, D, color=COLOR_LIST[i % len(COLOR_LIST)], linewidth=2,
                label=rf"$\alpha={a}$")

    ax.axhline(y=1.0, color=COLORS["grey"], linestyle="--", linewidth=1, label="No distortion")
    ax.set_xlabel("Refusal fraction $f$")
    ax.set_ylabel("Distortion ratio $D(f)$")
    ax.set_title("Distortion vs Refusal Fraction")
    ax.legend()
    fig.tight_layout()

    return _save(fig, out_dir, "distortion_ratio")


# ─── 10. Control Direction Comparison ─────────────────────────────────────────

def plot_control_comparison(safety_scan_path: str, control_scan_paths: dict,
                            out_dir: Path) -> Path:
    """Fig 5: Safety vs Control directions cumulative energy comparison.

    Parameters
    ----------
    safety_scan_path : str
        Path to safety scan energy_map.json.
    control_scan_paths : dict
        {"Language": "path/to/control_language/energy_map.json",
         "Code": "path/to/control_code/energy_map.json",
         "Math": "path/to/control_math/energy_map.json"}
    out_dir : Path
        Output directory for the figure.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    _setup_style()

    # Style per direction
    styles = {
        "Safety (RLHF)": {"color": "#d62728", "lw": 2.5, "ls": "-"},
        "Language":       {"color": "#1f77b4", "lw": 1.5, "ls": "--"},
        "Code":           {"color": "#2ca02c", "lw": 1.5, "ls": "-."},
        "Math":           {"color": "#9467bd", "lw": 1.5, "ls": ":"},
    }

    def load_cumulative_energy(scan_path):
        """Load scan JSON, compute per-layer total energy, return cumulative %."""
        with open(scan_path) as f:
            data = json.load(f)
        n_layers = data.get("num_layers",
                            max(h["layer"] for h in data["heads"]) + 1)
        layer_energy = np.zeros(n_layers)
        for h in data["heads"]:
            layer_energy[h["layer"]] += h.get("energy", 0)
        total = layer_energy.sum()
        if total > 0:
            layer_energy = layer_energy / total * 100
        return np.arange(n_layers), np.cumsum(layer_energy)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Safety
    layers, cum = load_cumulative_energy(safety_scan_path)
    s = styles["Safety (RLHF)"]
    ax.plot(layers, cum, color=s["color"], lw=s["lw"], ls=s["ls"],
            label="Safety (RLHF)")

    # Controls
    for label, path in control_scan_paths.items():
        layers_c, cum_c = load_cumulative_energy(path)
        s = styles.get(label, {"color": COLORS["grey"], "lw": 1.5, "ls": "--"})
        ax.plot(layers_c, cum_c, color=s["color"], lw=s["lw"], ls=s["ls"],
                label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative Energy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_title("Energy Distribution: Safety vs. Capability Directions")
    fig.tight_layout()

    return _save(fig, out_dir, "control_comparison")


# ─── CLI Entry ───────────────────────────────────────────────────────────────

def main():
    """CLI entry point for figure generation."""
    parser = argparse.ArgumentParser(
        prog="scalpel.plot",
        description="Generate SCALPEL publication figures.",
    )
    parser.add_argument("--mode", default="theory",
                        choices=["theory", "scan", "all"],
                        help="'theory' = no GPU needed; 'scan' = requires energy_map; 'all' = both")
    parser.add_argument("--scan-data", type=str, default=None,
                        help="Path to energy_map.json (required for scan/all modes)")
    parser.add_argument("--out-dir", type=str, default="figures",
                        help="Output directory for figures (default: figures/)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # Theory-only plots (no data needed)
    if args.mode in ("theory", "all"):
        from .adaptive_alpha import theoretical_curve
        # Standalone theoretical curve
        _setup_style()
        sel_arr, alpha_arr = theoretical_curve()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sel_arr, alpha_arr, color=COLORS["red"], linewidth=2)
        ax.axhline(y=2.0, color=COLORS["grey"], linestyle="--", linewidth=1, label=r"$\alpha=2$ (empirical)")
        ax.axhline(y=1.0, color=COLORS["grey"], linestyle=":", linewidth=1, label=r"$\alpha=1$")
        ax.set_xlabel("Selectivity")
        ax.set_ylabel(r"$\alpha^* = 1 + 1/\mathrm{Sel}^2$")
        ax.set_title("Theory-Guided Adaptive Alpha")
        ax.set_ylim(0.8, 5)
        ax.legend()
        fig.tight_layout()
        _save(fig, out_dir, "theory_adaptive_alpha")

        # Distortion ratio
        plot_distortion_ratio(out_dir)

    # Scan-based plots (need energy_map.json)
    if args.mode in ("scan", "all"):
        if args.scan_data is None:
            print("[PLOT] ERROR: --scan-data required for scan/all mode.")
            return

        scan_path = Path(args.scan_data)
        if not scan_path.exists():
            print(f"[PLOT] ERROR: {scan_path} not found.")
            return

        with open(scan_path) as f:
            energy_map = json.load(f)

        plot_selectivity_heatmap(energy_map, out_dir)
        plot_energy_distribution(energy_map, out_dir)
        plot_adaptive_alpha(energy_map, out_dir)

    print(f"[PLOT] Done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
