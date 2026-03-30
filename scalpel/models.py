"""Model registry and loading utilities.

Provides auto-detection of model architecture parameters from a local
HuggingFace ``config.json``, a runtime ``MODELS`` registry that maps
short names to architecture metadata, and helper functions for extracting
individual K / V projection weight matrices -- including support for
Grouped-Query Attention (GQA) models that fuse Q, K, and V into a single
``qkv_proj`` weight tensor.

Typical usage::

    name, info = resolve_model("/path/to/llama-model")
    model, tokenizer = load_model(name)
"""

import json as _json
import time
from pathlib import Path as _Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Model Registry ─────────────────────────────────────────────────────────

MODELS = {}  # Populated at runtime by resolve_model() from model path


# ─── Auto-detection from config.json ─────────────────────────────────────────

_TEMPLATE_MAP = {
    "llama": "llama3",
    "qwen2": "chatml",
    "phi3": "phi3",
    "phi": "phi3",
    "gemma": "gemma",
    "gemma2": "gemma",
    "mistral": "llama3",
}


def _read_model_config(path: str) -> dict:
    """Read and parse ``config.json`` from a model directory.

    Args:
        path: Filesystem path to a HuggingFace model directory.

    Returns:
        Parsed JSON content of ``config.json`` as a dictionary.

    Raises:
        FileNotFoundError: If ``config.json`` does not exist at *path*.
    """
    cfg_path = _Path(path) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found at {cfg_path}")
    with open(cfg_path) as f:
        return _json.load(f)


def _detect_template(model_type: str) -> str:
    """Map a HuggingFace ``model_type`` string to a SCALPEL template name.

    Uses the module-level ``_TEMPLATE_MAP`` lookup table.  If *model_type*
    is not present in the map the function falls back to ``"auto"``, which
    lets the tokenizer's built-in chat template handle formatting.

    Args:
        model_type: The ``model_type`` value from a HuggingFace
            ``config.json`` (e.g. ``"llama"``, ``"qwen2"``).

    Returns:
        A SCALPEL template identifier string (e.g. ``"llama3"``,
        ``"chatml"``, ``"auto"``).
    """
    return _TEMPLATE_MAP.get(model_type, "auto")


def _auto_detect_info(path: str) -> tuple:
    """Extract architecture params from a model's ``config.json``.

    Args:
        path: Filesystem path to a HuggingFace model directory containing
            a ``config.json`` file.

    Returns:
        A ``(info_dict, model_type_str)`` tuple where *info_dict* contains
        the keys ``path``, ``num_layers``, ``num_kv_heads``,
        ``num_attention_heads``, ``hidden_size``, and ``template``, and
        *model_type_str* is the raw ``model_type`` value from the config.
    """
    cfg = _read_model_config(path)

    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    num_attention_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    num_kv_heads = cfg.get("num_key_value_heads", num_attention_heads)
    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd")

    if num_layers is None or num_attention_heads is None or hidden_size is None:
        raise ValueError(
            f"Cannot detect architecture from {path}/config.json. "
            f"Got num_layers={num_layers}, num_attention_heads={num_attention_heads}, "
            f"hidden_size={hidden_size}"
        )

    model_type = cfg.get("model_type", "")
    template = _detect_template(model_type)

    info = {
        "path": str(_Path(path).resolve()),
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_attention_heads": num_attention_heads,
        "hidden_size": hidden_size,
        "template": template,
    }
    return info, model_type


def _path_to_short_name(path: str) -> str:
    """Derive a directory-safe short name from the last path component.

    Args:
        path: Filesystem path to a model directory.

    Returns:
        The final component of the resolved absolute path (e.g.
        ``"Llama-2-7b-chat-hf"`` for ``"/models/Llama-2-7b-chat-hf"``).
    """
    return _Path(path).resolve().name


def resolve_model(name_or_path: str) -> tuple:
    """Resolve a model short name or filesystem path.

    Returns ``(canonical_name, info_dict)``. If the input is a path,
    auto-detects architecture params and registers the model into
    ``MODELS``.

    Raises:
        ValueError: If *name_or_path* is neither a registered short name
            nor a valid directory containing ``config.json``.
    """
    # Already a registered short name
    if name_or_path in MODELS:
        return name_or_path, MODELS[name_or_path]

    # Try as a filesystem path
    p = _Path(name_or_path)
    if p.is_dir() and (p / "config.json").exists():
        info, model_type = _auto_detect_info(str(p))
        short_name = _path_to_short_name(str(p))
        if short_name in MODELS and MODELS[short_name]["path"] != info["path"]:
            short_name = short_name + "-auto"
        MODELS[short_name] = info
        print(
            f"[AUTO] Detected: {model_type or '?'} | "
            f"{info['num_layers']}L × {info['num_attention_heads']}Q/"
            f"{info['num_kv_heads']}KV × {info['hidden_size']}d | "
            f"template={info['template']}"
        )
        return short_name, info

    # Neither registered name nor valid path
    raise ValueError(
        f"'{name_or_path}' is not a valid model path.\n"
        f"Provide a local HuggingFace model directory containing config.json."
    )


def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer.

    Accepts either a registered short name or a filesystem path (resolved
    via :func:`resolve_model`).  Callers that also need
    ``MODELS[model_name]`` should call :func:`resolve_model` themselves
    first to obtain the canonical short name.

    Args:
        model_name: A registered model short name or a filesystem path to
            a HuggingFace model directory.
        device: Device map strategy passed to
            ``AutoModelForCausalLM.from_pretrained`` (default ``"auto"``).

    Returns:
        A ``(model, tokenizer)`` tuple where *model* is an
        ``AutoModelForCausalLM`` in eval mode on float16, and *tokenizer*
        is the corresponding ``AutoTokenizer``.
    """
    model_name, info = resolve_model(model_name)
    print(f"[LOAD] Loading {model_name} from {info['path']}...")
    t0 = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        info["path"], device_map=device, torch_dtype=torch.float16,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Llama-2 tokenizers shipped before the chat_template standard was
    # adopted, so they lack a Jinja2 chat_template attribute.  We inject a
    # standard Llama-2 [INST] / <<SYS>> template so that
    # tokenizer.apply_chat_template() works uniformly for all models.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        print(f"[LOAD] Patched chat_template (Llama-2 format)")
    
    print(f"[LOAD] Done in {time.time()-t0:.1f}s")
    return model, tokenizer


def free_model(model):
    """Release GPU memory held by *model*.

    Args:
        model: A PyTorch model whose references should be deleted before
            calling ``torch.cuda.empty_cache()``.
    """
    del model
    torch.cuda.empty_cache()


def get_W_V(attn_module):
    """Extract the V (value) projection weight matrix from an attention module.

    Handles both the common ``v_proj`` attribute (separate projections) and
    the fused ``qkv_proj`` attribute used by some architectures (e.g. Phi-3).

    For fused ``qkv_proj`` with Grouped-Query Attention (GQA) the weight
    rows are laid out as ``[Q_dim | K_dim | V_dim]`` where::

        Q_dim = num_heads     * head_dim
        K_dim = num_kv_heads  * head_dim
        V_dim = num_kv_heads  * head_dim

    Args:
        attn_module: A PyTorch attention sub-module that exposes either a
            ``v_proj`` or ``qkv_proj`` weight attribute.

    Returns:
        A ``torch.Tensor`` (float32) containing the V projection weights
        with shape ``(V_dim, hidden_size)``.

    Raises:
        ValueError: If the module has neither ``v_proj`` nor ``qkv_proj``.
    """
    if hasattr(attn_module, "v_proj"):
        return attn_module.v_proj.weight.data.float()
    elif hasattr(attn_module, "qkv_proj"):
        qkv = attn_module.qkv_proj.weight.data.float()
        num_heads = getattr(attn_module, "num_heads", getattr(attn_module, "num_key_value_heads"))
        # fused qkv_proj layout: [Q_dim | K_dim | V_dim]
        # Q_dim = num_heads * head_dim; K_dim = V_dim = num_kv_heads * head_dim
        q_dim = num_heads * attn_module.head_dim
        kv_dim = attn_module.num_key_value_heads * attn_module.head_dim
        return qkv[q_dim + kv_dim:]  # V slice starts after Q and K
    raise ValueError(f"Cannot find V projection in {type(attn_module)}")


def get_W_K(attn_module):
    """Extract the K (key) projection weight matrix from an attention module.

    Handles both the common ``k_proj`` attribute (separate projections) and
    the fused ``qkv_proj`` attribute used by some architectures (e.g. Phi-3).

    For fused ``qkv_proj`` with Grouped-Query Attention (GQA) the weight
    rows are laid out as ``[Q_dim | K_dim | V_dim]`` where::

        Q_dim = num_heads     * head_dim
        K_dim = num_kv_heads  * head_dim
        V_dim = num_kv_heads  * head_dim

    Args:
        attn_module: A PyTorch attention sub-module that exposes either a
            ``k_proj`` or ``qkv_proj`` weight attribute.

    Returns:
        A ``torch.Tensor`` (float32) containing the K projection weights
        with shape ``(K_dim, hidden_size)``.

    Raises:
        ValueError: If the module has neither ``k_proj`` nor ``qkv_proj``.
    """
    if hasattr(attn_module, "k_proj"):
        return attn_module.k_proj.weight.data.float()
    elif hasattr(attn_module, "qkv_proj"):
        qkv = attn_module.qkv_proj.weight.data.float()
        num_heads = getattr(attn_module, "num_heads", getattr(attn_module, "num_key_value_heads"))
        # fused qkv_proj layout: [Q_dim | K_dim | V_dim]
        # Q_dim = num_heads * head_dim; K_dim = V_dim = num_kv_heads * head_dim
        q_dim = num_heads * attn_module.head_dim
        kv_dim = attn_module.num_key_value_heads * attn_module.head_dim
        return qkv[q_dim: q_dim + kv_dim]  # K slice sits between Q and V
    raise ValueError(f"Cannot find K projection in {type(attn_module)}")


def format_prompt(prompt: str, model_name: str, system_prompt: str = None, no_sys: bool = False) -> list:
    """Format a user prompt as a chat-message list for ``tokenizer.apply_chat_template``.

    Args:
        prompt: The user's raw prompt text.
        model_name: A registered SCALPEL model short name (must already
            exist in ``MODELS``).
        system_prompt: Optional system-level instruction.  When *None* and
            a system message is allowed, a generic "You are a helpful
            assistant." default is used.
        no_sys: If ``True``, unconditionally omit the system message
            regardless of template support.

    Returns:
        A list of message dicts suitable for
        ``tokenizer.apply_chat_template`` (e.g.
        ``[{"role": "system", ...}, {"role": "user", ...}]``).
    """
    info = MODELS.get(model_name, {})
    template = info.get("template", "")

    # Models whose chat format has no system-role slot.  The tuple is
    # checked with ``in`` so partial matches work (e.g. "gemma" matches
    # templates "gemma" and "gemma2").
    no_system_models = ("gemma",)
    # "auto" template: include system message by default (tokenizer handles format)
    skip_sys = no_sys or template in no_system_models
    
    messages = []
    if not skip_sys and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif not skip_sys:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": prompt})
    return messages
