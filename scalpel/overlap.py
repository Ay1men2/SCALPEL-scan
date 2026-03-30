"""
Overlap checker — detect similarity between scan contrast pairs and test datasets.

Ensures that attack evaluation prompts are not contaminated by prompts used
during refusal direction extraction. Uses SequenceMatcher for pairwise comparison.

Usage:
    python -m scalpel.overlap --dataset harmbench
    python -m scalpel.overlap --dataset advbench --threshold 0.5
"""

import argparse
import ast
import re
from difflib import SequenceMatcher
from pathlib import Path


def _extract_contrast_pairs() -> list[tuple[str, str]]:
    """Extract CONTRAST_PAIRS from scan.py source without importing it.

    Avoids heavy numpy/torch dependency by parsing the source file
    directly and evaluating the list literal with ``ast.literal_eval``.

    Returns:
        List of (harmful_prompt, harmless_prompt) tuples parsed from
        the ``CONTRAST_PAIRS`` constant in ``scan.py``.

    Raises:
        RuntimeError: If ``CONTRAST_PAIRS`` cannot be located or its
            closing bracket is missing in the source file.
    """
    scan_path = Path(__file__).parent / "scan.py"
    source = scan_path.read_text()

    # Find the CONTRAST_PAIRS = [...] block
    match = re.search(
        r'^CONTRAST_PAIRS\s*=\s*\[',
        source,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Cannot find CONTRAST_PAIRS in scan.py")

    # Find matching closing bracket
    start = match.start()
    depth = 0
    for i, ch in enumerate(source[start:], start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                block = source[start:i + 1]
                break
    else:
        raise RuntimeError("Cannot find closing bracket for CONTRAST_PAIRS")

    # Strip the assignment prefix and parse as literal
    list_str = block.split('=', 1)[1].strip()
    # Remove comment lines
    lines = list_str.split('\n')
    cleaned = '\n'.join(
        line for line in lines if not line.strip().startswith('#')
    )
    pairs = ast.literal_eval(cleaned)
    return pairs


def get_scan_prompts() -> list[str]:
    """Extract harmful prompts from CONTRAST_PAIRS.

    Returns:
        List of harmful prompt strings (the first element of each pair).
    """
    pairs = _extract_contrast_pairs()
    return [harmful for harmful, _ in pairs]


def _load_behaviors_lightweight(dataset_name: str) -> list[dict]:
    """Load behaviors from data layer without heavy imports.

    Args:
        dataset_name: Name of the dataset to load (e.g., ``"harmbench"``,
            ``"advbench"``).

    Returns:
        List of behavior dicts, each containing at least a ``"prompt"`` key.
    """
    from .data import load_behaviors
    return load_behaviors(dataset_name)


def check_overlap(dataset_name: str, threshold: float = 0.6) -> list[dict]:
    """Check pairwise similarity between scan contrast prompts and test dataset.

    Args:
        dataset_name: Name of the test dataset (e.g., ``"harmbench"``,
            ``"advbench"``).
        threshold: Minimum ``SequenceMatcher`` ratio to flag a pair.
            Defaults to ``0.6``.

    Returns:
        List of dicts for every pair exceeding the threshold. Each dict
        contains keys ``scan_idx``, ``scan_prompt``, ``test_idx``,
        ``test_prompt``, and ``similarity``.
    """
    scan_prompts = get_scan_prompts()
    behaviors = _load_behaviors_lightweight(dataset_name)
    test_prompts = [b["prompt"] for b in behaviors]

    flagged = []
    for i, sp in enumerate(scan_prompts):
        for j, tp in enumerate(test_prompts):
            ratio = SequenceMatcher(None, sp.lower(), tp.lower()).ratio()
            if ratio > threshold:
                flagged.append({
                    "scan_idx": i,
                    "scan_prompt": sp,
                    "test_idx": j,
                    "test_prompt": tp,
                    "similarity": round(ratio, 4),
                })

    return flagged


def main():
    """CLI entry point for overlap checking."""
    parser = argparse.ArgumentParser(
        description="Check overlap between scan contrast pairs and test datasets"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name to check against")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Similarity threshold for flagging (default: 0.6)")
    args = parser.parse_args()

    scan_prompts = get_scan_prompts()
    print(f"[OVERLAP] Checking scan pairs vs {args.dataset} (threshold={args.threshold})")
    print(f"[OVERLAP] Scan prompts: {len(scan_prompts)}")

    flagged = check_overlap(args.dataset, args.threshold)

    if flagged:
        print(f"\n[OVERLAP] FLAGGED {len(flagged)} pairs above threshold:")
        for f in flagged:
            print(f"  sim={f['similarity']:.2f}  scan[{f['scan_idx']}]: {f['scan_prompt'][:60]}...")
            print(f"          test[{f['test_idx']}]: {f['test_prompt'][:60]}...")
    else:
        print(f"\n[OVERLAP] No overlaps found above threshold {args.threshold}")

    return flagged


if __name__ == "__main__":
    main()
