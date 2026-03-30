# SCALPEL-scan

**Scan component of SCALPEL: Safety-Circumventing Attack via Layer-wise Projection Erasure of Latents**

**Paper:** [SCALPEL: Bypassing LLM Safety Through the Unguarded KV Cache](https://zenodo.org/records/18625640)

**Author:** Tianyu Lu (Independent Researcher)

**Status:** Under review at [WOOT '26](https://www.usenix.org/conference/woot26)

---

## What This Repo Contains

This repository contains the **Scan** (Phase 1) and **analysis** components of SCALPEL. It enables researchers to:

- **Extract refusal directions** from any HuggingFace causal LM using 33 contrastive activation pairs
- **Compute per-head Energy and Selectivity maps** that identify which attention heads encode safety alignment
- **Visualize the "iceberg structure"** — safety alignment is concentrated in a small fraction of attention heads (typically <10% hold >50% of refusal energy)
- **Generate erasure plans** — head selection and alpha assignment based on selectivity scores
- **Compare safety vs. capability directions** using control contrast pairs (language, code, math)
- **Evaluate model outputs** using HarmBench CLS classifier or GPT-4o LLM-as-Judge
- **Compute statistical significance** with bootstrap confidence intervals across multiple runs

## What Is NOT Included

The **attack components** (Strike phase — KV cache erasure at inference time) are **withheld** per responsible disclosure:

- `strike.py` — KV cache projection erasure during inference
- Inference framework integrations (vLLM, TGI, SGLang, TensorRT-LLM, DeepSpeed)
- Capability benchmarks under attack conditions (MMLU, GSM8K, TruthfulQA with erasure)
- Arditi et al. weight abliteration baseline reproduction
- Batch experiment orchestrator

**Release timeline:** Attack code will be released after cache integrity defenses are available in at least one major inference framework. Responsible disclosure is ongoing.

## Responsible Disclosure

SCALPEL was responsibly disclosed to **5 inference frameworks** and **5 model vendors** prior to submission:

| Entity | Status |
|--------|--------|
| vLLM | [GHSA-vrvq-cg9c-5jp7](https://github.com/vllm-project/vllm/security/advisories/GHSA-vrvq-cg9c-5jp7) (closed, out of scope) |
| NVIDIA (TensorRT-LLM) | PSIRT Case #5963972 (under review) |
| HuggingFace (TGI) | Notified |
| SGLang | Notified |
| DeepSpeed (Microsoft) | MSRC VULN-056836 |
| Meta | Notified |
| Alibaba (Qwen) | Notified |
| Google (Gemma) | Notified |
| Mistral AI | Notified |
| DeepSeek | Notified |

## Installation

Requires Python >= 3.10 and a CUDA GPU.

```bash
pip install -e .
```

## Usage

### Scan a model

Extract refusal directions and compute the per-head energy/selectivity map:

```bash
python -m scalpel scan --model /path/to/Meta-Llama-3-8B-Instruct
```

Outputs:
- `outputs/scans/{model}/refusal_directions.pt` — per-layer refusal direction tensors
- `outputs/scans/{model}/energy_map.json` — per-head Energy, Baseline, Selectivity, Z-score

### Generate an erasure plan

Select target heads and assign alpha values based on scan results:

```bash
python -m scalpel plan --model /path/to/Meta-Llama-3-8B-Instruct --mode full_uniform --alpha 2.0
```

Six targeting modes available: `full_uniform`, `targeted_uniform`, `targeted_adaptive`, `full_adaptive`, `full_adaptive_clipped`, `full_depth_adaptive`.

### Visualize results

Generate publication-quality figures (no GPU required):

```bash
python -m scalpel.plot --mode all --scan-data outputs/scans/Meta-Llama-3-8B-Instruct/energy_map.json --out-dir figures/
```

### Control direction comparison (E9)

Scan with capability-specific contrast pairs to verify that safety directions are distinct from general capability directions:

```bash
# Safety scan (default)
python -m scalpel scan --model /path/to/Model

# Control scans
python -m scalpel scan --model /path/to/Model --pairs-file data/control_pairs/language.json --scan-tag Model-ctrl-language
python -m scalpel scan --model /path/to/Model --pairs-file data/control_pairs/code.json --scan-tag Model-ctrl-code
python -m scalpel scan --model /path/to/Model --pairs-file data/control_pairs/math.json --scan-tag Model-ctrl-math
```

### Check train/test overlap

Verify no contamination between scan contrast pairs and evaluation datasets:

```bash
python -m scalpel.overlap --dataset harmbench
```

### Evaluate outputs

Classify model outputs using HarmBench CLS (GPU) or GPT-4o Judge (API):

```bash
# HarmBench official classifier
python -m scalpel.eval_harmbench_cls --dir outputs/strikes/Model/TIMESTAMP/

# GPT-4o LLM-as-Judge
export OPENAI_API_KEY=sk-...
python -m scalpel judge --dir outputs/strikes/Model/TIMESTAMP/
```

## Key Formulas

- **Energy(l,h)** = ||W_V[l,h] @ r_l|| — how strongly head (l,h) responds to the refusal direction
- **Baseline(l,h)** = mean(||W_V[l,h] @ v_rand||) — average response to random directions
- **Selectivity(l,h)** = Energy / Baseline — values >> 1 indicate true safety-specific heads
- **Adaptive alpha:** alpha*(l,h) = 1 + 1/Selectivity(l,h)^2 — theory-guided erasure strength

## Model-Specific Notes

- **DeepSeek models:** Use `--no-sys` (no system prompt support)
- **Gemma models:** System prompt auto-skipped based on template detection
- **Phi-3:** Uses fused `qkv_proj` — handled transparently by `get_W_V()`/`get_W_K()`
- **70B models:** Auto-sharded across GPUs via `device_map="auto"`

## Adding Datasets

Registry-driven: add a JSONL file + one entry in `data/registry.yaml`. No code changes needed.

JSONL schema: `{"id": str, "prompt": str, "category": str, "source": str, "meta": {}}`

## Citation

```bibtex
@misc{lu2025scalpel,
  title     = {SCALPEL: Bypassing LLM Safety Through the Unguarded KV Cache},
  author    = {Tianyu Lu},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18625640},
  url       = {https://zenodo.org/records/18625640},
  note      = {Under review at WOOT '26}
}
```

## License

MIT
