"""
SCALPEL-scan CLI — Scan and analysis components.

Usage:
    python -m scalpel scan    --model /path/to/Model-Name
    python -m scalpel plan    --model /path/to/Model-Name --alpha 2.0
    python -m scalpel eval    --dir outputs/strikes/Model-Name/20260213_001234/
    python -m scalpel judge   --dir outputs/strikes/Model-Name/20260214_153042/

Note: Attack components (strike, run, bench, arditi) are withheld per
responsible disclosure. See README.md for details.
"""

import argparse
from pathlib import Path

# Base output directory
BASE_DIR = Path("outputs")
SCAN_DIR = BASE_DIR / "scans"
PLAN_DIR = BASE_DIR / "plans"

# Shared choices for plan subcommand
MODE_CHOICES = [
    "full_uniform", "targeted_uniform", "targeted_adaptive",
    "full_adaptive", "full_adaptive_clipped", "full_depth_adaptive",
]
DECAY_CHOICES = ["linear", "cosine", "halflife", "late_focused"]


def cmd_scan(args):
    """Handler for 'scalpel scan' subcommand."""
    from .models import resolve_model
    args.model = resolve_model(args.model)[0]
    from .scan import run_scan
    run_scan(args.model, SCAN_DIR, force=args.force, skip_extract=args.skip_extract,
             n_pairs=args.n_pairs, pairs_file=args.pairs_file,
             scan_tag=args.scan_tag, system_prompt=args.system_prompt,
             no_sys=args.no_sys)


def cmd_plan(args):
    """Handler for 'scalpel plan' subcommand."""
    from .models import resolve_model
    args.model = resolve_model(args.model)[0]
    from .plan import run_plan
    run_plan(
        args.model, SCAN_DIR, PLAN_DIR,
        mode=args.mode, alpha=args.alpha,
        z_threshold=args.z_threshold, gamma=args.gamma, cap=args.cap,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        decay=args.decay,
        scan_name=args.scan_name,
    )


def cmd_eval(args):
    """Handler for 'scalpel eval' subcommand."""
    from .eval import run_eval
    run_dir = Path(args.dir)
    run_eval(run_dir)


def cmd_judge(args):
    """Handler for 'scalpel judge' subcommand."""
    from .judge import run_judge
    run_judge(
        run_dir=args.dir,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.workers,
        strip_think=not args.no_strip_think,
    )


def main():
    """CLI entry point. Parses args and dispatches to subcommand handler."""
    parser = argparse.ArgumentParser(
        prog="scalpel",
        description="SCALPEL-scan: Safety-critical head analysis for LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ─── Common args ───
    from .data import list_datasets
    dataset_choices = list_datasets()
    model_help = "Path to local HuggingFace model directory (must contain config.json)"

    # ─── scan ───
    p_scan = sub.add_parser("scan", help="Extract refusal directions and compute energy map")
    p_scan.add_argument("--model", required=True, help=model_help)
    p_scan.add_argument("--force", action="store_true", help="Re-run even if results exist")
    p_scan.add_argument("--skip-extract", action="store_true", help="Skip direction extraction, only compute energy map")
    p_scan.add_argument("--n-pairs", type=int, default=None, help="Use only first N contrast pairs (default: all 33)")
    p_scan.add_argument("--pairs-file", type=str, default=None, help="JSON file with custom contrast pairs (for control directions)")
    p_scan.add_argument("--scan-tag", type=str, default=None, help="Override output directory name (default: model name)")
    p_scan.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt (default: 'You are a helpful assistant.')")
    p_scan.add_argument("--no-sys", action="store_true", help="No system prompt during scan")
    p_scan.set_defaults(func=cmd_scan)

    # ─── plan ───
    p_plan = sub.add_parser("plan", help="Generate erasure plan from scan results")
    p_plan.add_argument("--model", required=True, help=model_help)
    p_plan.add_argument("--mode", default="full_uniform", choices=MODE_CHOICES,
                        help="Plan mode. targeted_adaptive, full_adaptive, "
                             "full_adaptive_clipped are experimental")
    p_plan.add_argument("--alpha", type=float, default=2.0, help="Base alpha (default: 2.0)")
    p_plan.add_argument("--z-threshold", type=float, default=2.0, help="Z-score threshold for targeted modes")
    p_plan.add_argument("--gamma", type=float, default=0.5, help="Power-law exponent for adaptive mode")
    p_plan.add_argument("--cap", type=float, default=3.0, help="Max alpha multiplier for adaptive mode")
    p_plan.add_argument("--alpha-min", type=float, default=1.0, help="Min alpha for full_adaptive_clipped mode")
    p_plan.add_argument("--alpha-max", type=float, default=4.0, help="Max alpha for full_adaptive_clipped mode")
    p_plan.add_argument("--scan-name", type=str, default=None, help="Override scan directory name")
    p_plan.add_argument("--decay", type=str, default="linear", choices=DECAY_CHOICES,
                        help="Decay schedule for full_depth_adaptive mode (default: linear)")
    p_plan.set_defaults(func=cmd_plan)

    # ─── eval ───
    p_eval = sub.add_parser("eval", help="Evaluate a strike run")
    p_eval.add_argument("--dir", required=True, help="Path to strike run directory")
    p_eval.set_defaults(func=cmd_eval)

    # ─── judge (LLM-as-judge, runs locally via API) ───
    p_judge = sub.add_parser("judge", help="LLM-as-Judge scoring (GPT-4o, no GPU)")
    p_judge.add_argument("--dir", required=True, help="Path to strike run directory")
    p_judge.add_argument("--model", default="gpt-4o",
                         help="Judge model name (default: gpt-4o)")
    p_judge.add_argument("--api-key", default=None,
                         help="API key (or set OPENAI_API_KEY)")
    p_judge.add_argument("--workers", type=int, default=3,
                         help="Concurrent API requests (default: 3)")
    p_judge.add_argument("--no-strip-think", action="store_true",
                         help="Don't strip <think> blocks before judging")
    p_judge.set_defaults(func=cmd_judge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
