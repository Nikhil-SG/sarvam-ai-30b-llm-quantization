"""
Model architecture auto-detection for quantization research.

Introspects a loaded model to discover its structure (layers, projections,
MoE topology) without hardcoding any model-specific assumptions.

Produces a ``ModelArchInfo`` dict consumed by weight caching,
MSE heatmap, and visualization modules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from src.core.logger import get_logger

logger = get_logger(__name__)


def _is_linear_like_module(mod) -> bool:
    """Return True for float or quantized linear projection modules."""
    if isinstance(mod, nn.Linear):
        return True

    if hasattr(mod, "qweight"):
        return True

    weight = getattr(mod, "weight", None)
    if weight is not None:
        if hasattr(weight, "quant_state") or hasattr(weight, "SCB"):
            return True
        if all(hasattr(mod, attr) for attr in ("in_features", "out_features")):
            return True

    if hasattr(mod, "dequantize") and callable(mod.dequantize):
        return True

    return False


def detect_architecture(model) -> Dict[str, Any]:
    """
    Auto-detect model architecture from a loaded HuggingFace model.

    Returns a dict with:
        model_type          str     e.g. "sarvam_moe"
        num_layers          int     number of transformer layers
        hidden_size         int     hidden dimension
        is_moe              bool    whether the model uses MoE
        num_experts         int     experts per MoE layer (0 if dense)
        num_experts_per_tok int     active experts per token
        num_shared_experts  int     shared (always-active) experts
        first_dense_layer   int     first layer index that uses dense MLP
        total_params_b      float   total parameters in billions
        active_params_b     float   active params per token in billions (MoE)
        layer_prefix        str     dotted path to layer list
        attn_projections    list    attention projection names found
        mlp_projections     list    MLP projection names found (per layer type)
        moe_projections     list    MoE-specific projection names
        vocab_size          int     vocabulary size
    """
    config = model.config
    info: Dict[str, Any] = {}

    # Basic config attributes
    info["model_type"] = getattr(config, "model_type", "unknown")
    info["num_layers"] = getattr(config, "num_hidden_layers", 0)
    info["hidden_size"] = getattr(config, "hidden_size", 0)
    info["vocab_size"] = getattr(config, "vocab_size", 0)

    # MoE detection
    info["num_experts"] = getattr(config, "num_experts", 0) or getattr(config, "num_local_experts", 0)
    info["num_experts_per_tok"] = getattr(config, "num_experts_per_tok", 0) or getattr(config, "num_selected_experts", 0)
    info["num_shared_experts"] = getattr(config, "num_shared_experts", 0)
    info["is_moe"] = info["num_experts"] > 0
    info["first_dense_layer"] = getattr(config, "first_k_dense_replace", 0)

    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    info["total_params_b"] = round(total_params / 1e9, 2)

    # Estimate active params for MoE (rough: non-expert params + active expert params)
    if info["is_moe"] and info["num_experts"] > 0:
        expert_params = 0
        non_expert_params = 0
        for name, p in model.named_parameters():
            if _is_expert_param(name):
                expert_params += p.numel()
            else:
                non_expert_params += p.numel()
        # Active = non-expert + (expert_per_tok / total_experts) * expert_params
        active_ratio = info["num_experts_per_tok"] / info["num_experts"] if info["num_experts"] else 1.0
        active_params = non_expert_params + expert_params * active_ratio
        info["active_params_b"] = round(active_params / 1e9, 2)
    else:
        info["active_params_b"] = info["total_params_b"]

    # Layer prefix detection
    info["layer_prefix"] = _detect_layer_prefix(model)

    # Projection discovery (inspect actual named modules)
    attn_projs, mlp_projs, moe_projs = _discover_projections(model, info)
    info["attn_projections"] = attn_projs
    info["mlp_projections"] = mlp_projs
    info["moe_projections"] = moe_projs

    logger.info(f"Architecture: {info['model_type']}")
    logger.info(f"  Layers: {info['num_layers']}, Hidden: {info['hidden_size']}")
    logger.info(f"  Total params: {info['total_params_b']}B, Active: {info['active_params_b']}B")
    if info["is_moe"]:
        logger.info(
            f"  MoE: {info['num_experts']} experts, "
            f"top-{info['num_experts_per_tok']}, "
            f"{info['num_shared_experts']} shared"
        )
    logger.info(f"  Attn projections: {attn_projs}")
    logger.info(f"  MLP projections: {mlp_projs}")
    if moe_projs:
        logger.info(f"  MoE projections: {moe_projs}")

    return info


def get_target_layers_for_heatmap(arch_info: Dict[str, Any]) -> List[str]:
    """
    Generate the full list of (layer_name, projection) pairs for
    MSE heatmap analysis based on auto-detected architecture.

    For MoE models, includes attention projections for all layers,
    plus the shared expert projection for MoE layers.
    """
    prefix = arch_info["layer_prefix"]
    n_layers = arch_info["num_layers"]
    attn_projs = arch_info["attn_projections"]
    moe_projs = arch_info["moe_projections"]
    first_dense = arch_info.get("first_dense_layer", 0)

    layer_names = []
    for idx in range(n_layers):
        # Attention projections (present in all layers)
        for proj in attn_projs:
            layer_names.append(f"{prefix}.{idx}.{proj}")
        # MoE projections (only for MoE layers)
        if arch_info["is_moe"] and idx >= first_dense and moe_projs:
            for proj in moe_projs:
                layer_names.append(f"{prefix}.{idx}.{proj}")

    return layer_names


def _is_expert_param(name: str) -> bool:
    """Check if a parameter belongs to an expert (routed, not shared)."""
    # Match patterns like 'mlp.experts.42.down_proj'
    return bool(re.search(r"\.experts\.\d+\.", name))


def _detect_layer_prefix(model) -> str:
    """Find the dotted prefix to the ModuleList of decoder layers."""
    # Common patterns: model.layers, model.model.layers, transformer.h
    for candidate in ["model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = candidate.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return candidate
        except AttributeError:
            continue

    # Fallback: search named modules
    for name, module in model.named_modules():
        if name.endswith(".layers") and hasattr(module, "__len__") and len(module) > 1:
            return name

    return "model.layers"  # default assumption


def _is_shared_expert_projection(name: str, layer_prefix: str, layer_idx: int) -> bool:
    needle = f"{layer_prefix}.{layer_idx}.mlp.shared_expert"
    return needle in name


def _discover_projections(
    model, arch_info: Dict[str, Any]
) -> tuple[List[str], List[str], List[str]]:
    """
    Inspect the first few layers to discover projection module names.

    Returns:
        (attn_projections, mlp_projections, moe_projections)
    """
    prefix = arch_info["layer_prefix"]
    n_layers = arch_info["num_layers"]
    first_dense = arch_info.get("first_dense_layer", 0)

    attn_projs: Set[str] = set()
    mlp_projs: Set[str] = set()
    moe_projs: Set[str] = set()

    # Inspect layer 0 (often dense even in MoE models)
    _inspect_layer(model, f"{prefix}.0", attn_projs, mlp_projs)

    # Inspect a MoE layer if available
    if arch_info["is_moe"] and first_dense < n_layers:
        moe_layer_idx = first_dense  # first MoE layer
        _inspect_layer(model, f"{prefix}.{moe_layer_idx}", attn_projs, mlp_projs)

        # Extract shared expert projections specifically
        for name, mod in model.named_modules():
            if _is_shared_expert_projection(name, prefix, moe_layer_idx):
                # Get the relative path after the layer prefix
                rel = name.split(f"{prefix}.{moe_layer_idx}.")[-1]
                if _is_linear_like_module(mod):
                    moe_projs.add(rel)

    return sorted(attn_projs), sorted(mlp_projs), sorted(moe_projs)


def _inspect_layer(
    model, layer_path: str, attn_projs: Set[str], mlp_projs: Set[str]
) -> None:
    """Add discovered projection names from one layer."""
    for name, mod in model.named_modules():
        if not name.startswith(layer_path + "."):
            continue
        if not _is_linear_like_module(mod):
            continue

        # Relative path within the layer
        rel = name.split(layer_path + ".")[-1]

        # Skip expert-specific projections (too many to enumerate)
        if re.search(r"\.experts\.\d+\.", rel):
            continue

        # Classify by parent module type name
        if "attention" in rel or "self_attn" in rel or "attn" in rel:
            attn_projs.add(rel)
        elif "mlp" in rel or "feed_forward" in rel:
            mlp_projs.add(rel)
