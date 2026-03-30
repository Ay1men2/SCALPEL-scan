"""
SCALPEL dataset loader — registry-driven, schema-unified.

All datasets live as JSONL files under data/ and are declared in data/registry.yaml.
Adding a new dataset = add JSONL file + one registry entry. No code changes.

JSONL record schema:
    {"id": str, "prompt": str, "category": str, "source": str, "meta": {}}
"""

import json
from pathlib import Path

import yaml


# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
REGISTRY_PATH = DATA_DIR / "registry.yaml"


# ─── Registry Loading ───────────────────────────────────────────────────────

_registry_cache = None


def _load_registry() -> dict:
    """Load and cache the dataset registry from ``registry.yaml``.

    Returns:
        The parsed registry dictionary.  Subsequent calls return the
        cached result without re-reading the file.
    """
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    text = REGISTRY_PATH.read_text(encoding="utf-8")
    reg = yaml.safe_load(text)

    _registry_cache = reg
    return reg


# ─── JSONL Loading ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list:
    """Load a JSONL file and return its records as a list of dicts.

    Args:
        path: Filesystem path to a ``.jsonl`` file.  Each non-empty line
            must be valid JSON.

    Returns:
        A list of dictionaries, one per non-empty line in the file.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _apply_slice(records: list, slice_str: str) -> list:
    """Apply a Python-style slice string to a list of records.

    Args:
        records: The full list of record dicts to slice.
        slice_str: A colon-delimited slice expression such as ``":8"``,
            ``":20"``, or ``"10:50"``.  An empty string or bare ``":"``
            returns *records* unchanged.

    Returns:
        The sliced subset of *records*.
    """
    if not slice_str or slice_str == ":":
        return records
    parts = slice_str.split(":")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return records[start:end]


# ─── Public API ─────────────────────────────────────────────────────────────

def load_behaviors(name: str) -> list:
    """Load dataset as list of behavior dicts.

    Each dict: {"id": str, "prompt": str, "category": str, "source": str, "meta": {}}

    Supports:
      - Named datasets from registry (advbench, harmbench, benign)
      - Aliases (test8, advbench20, ...)
      - Slice syntax: "advbench:20" -- first 20 of advbench

    Raises:
        ValueError: If *name* is not a registered dataset, alias, or
            valid slice expression, or if the dataset uses the HuggingFace
            loader and must be invoked via ``python -m scalpel bench``.
        FileNotFoundError: If the resolved JSONL file does not exist on
            disk.
    """
    reg = _load_registry()
    aliases = reg.get("aliases", {})

    # Check slice syntax: "name:N"
    slice_str = None
    if ":" in name and name not in reg and name not in aliases:
        name, slice_str = name.rsplit(":", 1)
        slice_str = ":" + slice_str

    # Resolve alias
    if name in aliases:
        alias = aliases[name]
        real_name = alias["dataset"]
        slice_str = alias.get("slice", slice_str)
        return _apply_slice(load_behaviors(real_name), slice_str)

    # Direct dataset
    if name not in reg:
        available = list(reg.keys()) + list(aliases.keys())
        available = [k for k in available if k != "aliases"]
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    entry = reg[name]

    # HuggingFace datasets (capability benchmarks) are not loaded here
    if entry.get("loader") == "huggingface":
        raise ValueError(
            f"Dataset '{name}' uses HuggingFace loader. "
            f"Use 'python -m scalpel bench {name}' instead."
        )

    path = DATA_DIR / entry["path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = _load_jsonl(path)

    if slice_str:
        records = _apply_slice(records, slice_str)

    return records


def load_dataset(name: str) -> list:
    """Load dataset as list of prompt strings (backward compatible)."""
    return [b["prompt"] for b in load_behaviors(name)]


def list_datasets() -> list:
    """Return available dataset names (datasets + aliases).

    Returns:
        A list of strings containing all registered dataset names
        followed by all alias names.
    """
    reg = _load_registry()
    aliases = reg.get("aliases", {})
    names = [k for k in reg if k != "aliases"]
    names.extend(aliases.keys())
    return names
