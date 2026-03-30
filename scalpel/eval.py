"""
Phase 4: EVAL — Post-strike evaluation utilities.

Classification is handled by two dedicated evaluators:
  - HarmBench CLS   (eval_harmbench_cls.py) — local GPU classifier
  - GPT-4o Judge    (judge.py)              — LLM-as-Judge via API

This module provides:
  - extract_final_response(): strip R1 <think> blocks (used by CLS)
  - run_eval(): lightweight post-strike summary (output count + metadata)
"""

import json
import re
from pathlib import Path


# ─── R1 Think Block Extraction ──────────────────────────────────────────────

def extract_final_response(text: str) -> str:
    """Extract the final response from R1-style outputs with <think> blocks.

    R1 models may wrap reasoning in ``<think>...</think>`` before the final
    answer.  This function returns only the text *after* the last closing
    ``</think>`` tag.  If no think block is present the original text is
    returned unchanged.

    Args:
        text: Raw model output that may contain one or more
            ``<think>...</think>`` or ``<|think|>...</|think|>`` blocks
            followed by the final answer.

    Returns:
        The text after the last ``</think>`` (or ``</|think|>``) tag,
        stripped of leading/trailing whitespace.  Falls back to the
        original *text* if no think block is found or if nothing
        follows the closing tag.
    """
    # Handle both <think>...</think> and <|think|>...</|think|> variants
    pattern = r"</think>|</\|think\|>"
    parts = re.split(pattern, text, flags=re.DOTALL)
    if len(parts) > 1:
        final = parts[-1].strip()
        return final if final else text  # fallback to original if nothing after </think>
    return text


# ─── Lightweight Post-Strike Summary ────────────────────────────────────────

def run_eval(run_dir: Path):
    """Lightweight post-strike summary: count outputs, log metadata.

    Full classification is deferred to HarmBench CLS or GPT-4o Judge.
    This just saves a basic summary.json with output counts.

    Args:
        run_dir: Path to the experiment run directory.  Must contain an
            ``outputs.jsonl`` file with one JSON object per line.  The
            first record is used to infer model name, dataset, plan mode,
            and alpha_base for the summary.

    Returns:
        A dict with summary metadata (run_dir, model, dataset,
        plan_mode, alpha_base, total count, and a note), or ``None``
        if ``outputs.jsonl`` is missing or empty.  The same dict is
        persisted to ``<run_dir>/summary.json``.
    """
    run_dir = Path(run_dir)
    output_path = run_dir / "outputs.jsonl"
    if not output_path.exists():
        print(f"[EVAL] No outputs at {output_path}")
        return None

    results = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        print(f"[EVAL] Empty outputs at {output_path}")
        return None

    summary = {
        "run_dir": str(run_dir),
        "model": results[0].get("model", "unknown"),
        "dataset": results[0].get("dataset", "unknown"),
        "plan_mode": results[0].get("plan_mode", "unknown"),
        "alpha_base": results[0].get("alpha_base", 0),
        "total": len(results),
        "note": "Classification deferred to HarmBench CLS / GPT-4o Judge",
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[EVAL] {len(results)} outputs saved | {run_dir}")
    print(f"[EVAL] Next steps:")
    print(f"  HarmBench CLS:  python -m scalpel.eval_harmbench_cls --dir {run_dir}")
    print(f"  GPT-4o Judge:   python -m scalpel judge --dir {run_dir}")
    return summary
