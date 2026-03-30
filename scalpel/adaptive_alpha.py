"""
Theory-grounded adaptive α for SCALPEL.

Derives per-head erasure strength from the noise compensation model
(Appendix A) using Selectivity as a proxy for estimation precision.

Key result:  α*(l,h) = 1 + 1/Selectivity(l,h)²

Boundary conditions:
  Sel → ∞  ⟹  α → 1.0  (clean signal, no over-projection)
  Sel = 1  ⟹  α = 2.0  (matches empirical global optimum)
  Sel → 0  ⟹  α → ∞   (clamped in practice)

NOTE: Only ``alpha_from_selectivity()`` is used in the main pipeline
(called by ``plan.py`` for full_adaptive / full_adaptive_clipped modes).
The other two functions are retained for plotting and analysis.
"""

import numpy as np


def alpha_from_selectivity(selectivity: float) -> float:
    """Compute single-head erasure strength from the noise compensation model.

    Implements the closed-form result from Appendix A:
    ``alpha*(l,h) = 1 + 1 / Sel(l,h)^2``.

    Args:
        selectivity (float): Selectivity score for the head, as computed
            during the scan phase.  Values should be positive; a floor of
            0.1 is applied internally.

    Returns:
        float: The theory-derived alpha value for this head.
    """
    # Floor selectivity at 0.1 to prevent division by zero (or near-zero)
    # when a head has negligible selectivity.
    return 1.0 + 1.0 / max(selectivity, 0.1) ** 2


def compute_adaptive_alphas(head_infos: list, alpha_min: float = 1.0,
                            alpha_max: float = 4.0, clip: bool = False) -> dict:
    """Compute per-head α for all heads in scan result.

    Parameters
    ----------
    head_infos : list of dict
        Each dict must have: {"layer": int, "head": int, "selectivity": float}
    alpha_min, alpha_max : float
        Bounds for clipped mode.
    clip : bool
        If True, clamp α to [alpha_min, alpha_max].

    Returns
    -------
    dict with:
        "alphas": {(layer, head): float}
        "stats": {mean, median, std, min, max, n_heads}

    Raises
    ------
    This function does not raise.

    Note
    ----
    Currently unused in the main pipeline (dead code).  Retained for
    plotting, analysis notebooks, and potential future integration.
    """
    alphas = {}
    values = []
    for h in head_infos:
        sel = h.get("selectivity", 1.0)
        a = alpha_from_selectivity(sel)
        if clip:
            a = max(alpha_min, min(alpha_max, a))
        alphas[(h["layer"], h["head"])] = a
        values.append(a)

    values = np.array(values) if values else np.array([0.0])
    stats = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_heads": len(values),
    }
    return {"alphas": alphas, "stats": stats}


def theoretical_curve(sel_min: float = 0.3, sel_max: float = 20.0,
                      n_points: int = 200):
    """Generate the theoretical alpha-vs-selectivity curve for plotting.

    Evaluates ``alpha = 1 + 1/sel^2`` over a linearly spaced selectivity
    range, useful for overlaying the analytic curve on empirical scatter
    plots.

    Args:
        sel_min (float): Lower bound of the selectivity range.
        sel_max (float): Upper bound of the selectivity range.
        n_points (int): Number of evenly spaced evaluation points.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair ``(sel, alpha)`` where
        ``sel`` is the selectivity array and ``alpha`` is the corresponding
        theory-derived alpha array.
    """
    sel = np.linspace(sel_min, sel_max, n_points)
    alpha = 1.0 + 1.0 / sel ** 2
    return sel, alpha
