"""
Unified weight extraction, caching, and cross-format comparison.

Handles dequantisation from BF16, GPTQ (auto-gptq), and INT8,
and caches sampled weight vectors for later MSE / distribution analysis.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  WeightExtractor – pull float32 weights from any quantisation format
# ═══════════════════════════════════════════════════════════════════════════
class WeightExtractor:
    """Stateless helper to navigate a model and dequantise one layer."""

    # ── navigation ──────────────────────────────────────────────────────
    @staticmethod
    def get_module_by_name(model, layer_name: str):
        """Walk dotted path and return the leaf ``nn.Module``."""
        module = model
        for part in layer_name.split("."):
            if not hasattr(module, part):
                raise AttributeError(
                    f"'{type(module).__name__}' has no attribute '{part}' "
                    f"in path '{layer_name}'"
                )
            module = getattr(module, part)
        return module

    # ── public entry-point ──────────────────────────────────────────────
    @staticmethod
    def extract_weight(model, layer_name: str) -> torch.Tensor:
        """
        Return the effective float32 weight tensor (on CPU) for *any*
        supported quantisation back-end.

        Each path logs which dequantisation route was taken so silent
        fallthrough bugs (which previously produced near-BF16 values
        for quantised formats) are immediately visible.
        """
        module = WeightExtractor.get_module_by_name(model, layer_name)

        # 1) Standard float weight (BF16 / FP16 / FP32)
        if hasattr(module, "weight"):
            w = module.weight

            # bitsandbytes 4-bit  (Params4bit with .quant_state)
            if hasattr(w, "quant_state") and w.quant_state is not None:
                logger.debug(
                    f"  [{layer_name}] path: bnb_4bit "
                    f"(dtype={w.dtype}, quant_state={type(w.quant_state).__name__})"
                )
                return WeightExtractor._dequant_bnb_4bit(module)

            # bitsandbytes LLM.int8() — MUST be checked before the generic
            # dequantize() path below.  torch.Tensor inherits dequantize()
            # from PyTorch's quantization API; calling it on a plain int8
            # tensor (Int8Params) silently casts raw int8 values to float32
            # without applying any scale factor, producing values like ±127
            # instead of the correct ±1 range.  Route to _dequant_bnb_int8()
            # which applies the per-row SCB scale correctly.
            if isinstance(w, torch.Tensor) and w.dtype == torch.int8:
                logger.debug(f"  [{layer_name}] path: bnb_int8")
                return WeightExtractor._dequant_bnb_int8(module)

            # optimum-quanto QBytesTensor — genuinely overrides
            # dequantize() to return properly scaled float32.
            # Check for quanto-specific attributes to distinguish from
            # vanilla torch.Tensor.dequantize() which would silently
            # cast int8 → float without scaling.
            if hasattr(w, "_data") and hasattr(w, "_scale"):
                # Definitely a quanto QTensor — use _data * _scale directly
                # for reliability (avoids quanto internal dispatch issues).
                try:
                    result = (w._data.float() * w._scale.float()).detach().cpu()
                    logger.debug(
                        f"  [{layer_name}] path: quanto _data*_scale "
                        f"(_data.dtype={w._data.dtype}, "
                        f"_scale.shape={tuple(w._scale.shape)})"
                    )
                    return result
                except Exception as exc:
                    logger.warning(
                        f"  [{layer_name}] quanto _data*_scale FAILED: {exc} "
                        f"— trying w.dequantize()"
                    )
                    # Fall through to dequantize() call below

            # Generic dequantize() — only for tensor subclasses that
            # override it (quanto QTensor, etc.).  NOT safe for plain
            # int8/uint8 tensors.
            if (
                hasattr(w, "dequantize")
                and callable(w.dequantize)
                and w.dtype not in (
                    torch.int8, torch.uint8, torch.int16, torch.int32
                )
            ):
                try:
                    result = w.dequantize().detach().cpu().float()
                    logger.debug(
                        f"  [{layer_name}] path: w.dequantize() "
                        f"(dtype={w.dtype}, type={type(w).__name__})"
                    )
                    return result
                except Exception as exc:
                    logger.warning(
                        f"  [{layer_name}] w.dequantize() FAILED: {exc} "
                        f"— falling through to plain float path"
                    )

            # plain float dtype — ONLY for genuinely unquantised weights
            if isinstance(w, torch.Tensor) and w.dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                logger.debug(
                    f"  [{layer_name}] path: plain_float (dtype={w.dtype})"
                )
                return w.data.detach().cpu().float()

            # float8 raw tensor (NOT wrapped in quanto) — can happen if
            # a model is loaded from a raw float8 safetensors checkpoint.
            # This is NOT a proper dequantisation (no scale factor), so
            # log a WARNING.  The cast to float32 just widens the bits.
            if isinstance(w, torch.Tensor) and w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                logger.warning(
                    f"  [{layer_name}] path: RAW_FLOAT8_NO_SCALE "
                    f"(dtype={w.dtype}) — values may be inaccurate!"
                )
                return w.data.detach().cpu().float()

        # 2) HQQ quantised layers (HQQLinear)
        if hasattr(module, "dequantize") and callable(module.dequantize):
            try:
                result = module.dequantize().detach().cpu().float()
                logger.debug(f"  [{layer_name}] path: HQQ_module_dequantize")
                return result
            except Exception as exc:
                logger.warning(
                    f"  [{layer_name}] HQQLinear dequantize() FAILED: {exc}"
                )

        # 3) Packed integer weights (GPTQ-style and related layouts)
        if hasattr(module, "qweight"):
            logger.debug(f"  [{layer_name}] path: packed_int")
            return WeightExtractor._dequant_packed(module)

        raise ValueError(
            f"Cannot extract weights from '{layer_name}' "
            f"(type={type(module).__name__}). "
            f"Public attrs: {[a for a in dir(module) if not a.startswith('_')]}"
        )

    # ── bitsandbytes 4-bit ──────────────────────────────────────────────
    @staticmethod
    def _dequant_bnb_4bit(module) -> torch.Tensor:
        try:
            import bitsandbytes.functional as bnb_F

            w = module.weight
            qs = w.quant_state
            logger.debug(
                f"  bnb_4bit: w.data.shape={tuple(w.data.shape)}, "
                f"w.data.dtype={w.data.dtype}, "
                f"quant_type={getattr(qs, 'quant_type', '?')}, "
                f"blocksize={getattr(qs, 'blocksize', '?')}"
            )
            result = bnb_F.dequantize_4bit(w.data, qs).cpu().float()
            logger.debug(
                f"  bnb_4bit result: shape={tuple(result.shape)}, "
                f"range=[{result.min():.4f}, {result.max():.4f}], "
                f"std={result.std():.6f}"
            )
            return result
        except Exception as exc:
            logger.error(f"BnB 4-bit dequant failed: {exc}")
            raise

    # ── bitsandbytes INT8 (LLM.int8 / Linear8bitLt) ─────────────────────
    @staticmethod
    def _dequant_bnb_int8(module) -> torch.Tensor:
        """
        Dequantize a bitsandbytes LLM.int8() weight matrix.

        Strategies (tried in order):
          1. Use per-row scale factors (SCB) from weight param or module state.
          2. Pass an identity matrix through the Linear8bitLt module so
             bitsandbytes' own int8 matmul reconstructs the effective float
             weights, inclusive of outlier-column FP16 fallback.
          3. Last resort: raw int8 / 127 (approximate, LOSSY).
        """
        w = module.weight

        # ── Strategy 1: locate SCB from known sources ──────────────────
        scb = getattr(w, "SCB", None)

        if scb is None and hasattr(module, "state"):
            scb = getattr(module.state, "SCB", None)

        if scb is None:
            qs = getattr(w, "quant_state", None)
            if qs is not None:
                scb = getattr(qs, "absmax", getattr(qs, "SCB", None))

        if scb is not None:
            w_f = w.data.detach().cpu().float()
            scale = scb.detach().cpu().float()
            if scale.numel() == w_f.shape[0]:
                logger.debug("INT8 dequant via SCB scale factors")
                return w_f * (scale.unsqueeze(1) / 127.0)
            logger.warning(
                f"INT8 SCB shape mismatch: SCB {scale.numel()} vs "
                f"weight rows {w_f.shape[0]} — falling through"
            )

        # ── Strategy 2: identity-matrix forward pass ───────────────────
        #    y = W @ x + b.  With x = I (identity), y = W + b.
        #    Subtracting bias recovers W exactly, using bitsandbytes'
        #    own int8 matmul (which handles outlier columns in FP16).
        try:
            in_features = module.in_features
            device = w.device
            chunk_size = min(in_features, 1024)
            chunks = []

            for start in range(0, in_features, chunk_size):
                end = min(start + chunk_size, in_features)
                sz = end - start
                eye = torch.zeros(
                    sz, in_features, device=device, dtype=torch.float16
                )
                rows = torch.arange(sz, device=device)
                cols = torch.arange(start, end, device=device)
                eye[rows, cols] = 1.0

                with torch.no_grad():
                    out = module(eye)
                if module.bias is not None:
                    out = out - module.bias
                chunks.append(out.cpu().float())
                del eye, out

            # chunks: list of [chunk, out_features] → cat → transpose
            W_rec = torch.cat(chunks, dim=0).t()  # [out, in]
            logger.debug(
                f"INT8 dequant via identity-matrix forward "
                f"({in_features // chunk_size + 1} chunks)"
            )
            return W_rec
        except Exception as exc:
            logger.warning(f"INT8 identity-matrix dequant failed: {exc}")

        # ── Strategy 3: raw normalisation (approximate) ────────────────
        logger.warning(
            "INT8: all dequant strategies failed — using approximate ÷127"
        )
        w_f = w.data.detach().cpu().float()
        return w_f / 127.0

        # ── Packed integer weights ──────────────────────────────────────────
    @staticmethod
    def _dequant_packed(module) -> torch.Tensor:
        """
        Unpack int32-packed weights and apply scale / zero-point.

        Supports three packing layouts:
          1. GPTQ-style:          qweight [in//pack, out]
                    2. Row-packed style:    qweight [out, in//pack]
                    3. Transposed-pack:     qweight [in, out//pack]
                         (seen when in_features > out_features,
              e.g. mlp.down_proj in large MoE models: [28672, 1024] with
              scales [224, 8192])
        """
        try:
            qweight = module.qweight.data.cpu().to(torch.int32)
            scales = module.scales.data.cpu().float()
            has_qzeros = (
                hasattr(module, "qzeros")
                and module.qzeros is not None
            )
            qzeros = (
                module.qzeros.data.cpu().to(torch.int32) if has_qzeros else None
            )

            bits = getattr(module, "bits", getattr(module, "w_bit", 4))
            group_size = getattr(
                module, "group_size", getattr(module, "q_group_size", 128)
            )

            pack_num = 32 // bits
            mask = (1 << bits) - 1

            # scales → [num_groups, out_features]
            if scales.dim() != 2:
                raise ValueError(f"Unexpected scales shape: {scales.shape}")
            num_groups, out_features = scales.shape
            in_features = num_groups * group_size

            # ── detect packing orientation ──────────────────────────────
            if qweight.shape[1] == out_features:
                # Layout 1 — GPTQ-style  [in_features // pack_num, out_features]
                weight = torch.zeros(
                    in_features, out_features, dtype=torch.float32
                )
                for i in range(pack_num):
                    weight[i::pack_num, :] = (
                        (qweight >> (bits * i)) & mask
                    ).float()

                zeros = WeightExtractor._unpack_zeros(
                    qzeros, bits, pack_num, num_groups, out_features, scales
                )
                for g in range(num_groups):
                    s, e = g * group_size, min((g + 1) * group_size, in_features)
                    weight[s:e, :] = (
                        weight[s:e, :] - zeros[g : g + 1, :]
                    ) * scales[g : g + 1, :]

                return weight.T  # → [out, in]

            elif qweight.shape[0] == out_features:
                # Layout 2 — row-packed  [out_features, in_features // pack_num]
                weight = torch.zeros(
                    out_features, in_features, dtype=torch.float32
                )
                for i in range(pack_num):
                    weight[:, i::pack_num] = (
                        (qweight >> (bits * i)) & mask
                    ).float()

                zeros = WeightExtractor._unpack_zeros(
                    qzeros, bits, pack_num, num_groups, out_features, scales
                )
                for g in range(num_groups):
                    s, e = g * group_size, min((g + 1) * group_size, in_features)
                    weight[:, s:e] = (
                        weight[:, s:e] - zeros[g : g + 1, :].T
                    ) * scales[g : g + 1, :].T

                return weight  # already [out, in]

            elif (
                qweight.shape[0] == in_features
                and qweight.shape[1] * pack_num == out_features
            ):
                # Layout 3 — transposed-pack
                # qweight [in_features, out_features // pack_num]
                # Packing is along the output dimension.
                # Seen for layers where in > out (e.g. down_proj).
                weight = torch.zeros(
                    in_features, out_features, dtype=torch.float32
                )
                for i in range(pack_num):
                    weight[:, i::pack_num] = (
                        (qweight >> (bits * i)) & mask
                    ).float()

                zeros = WeightExtractor._unpack_zeros(
                    qzeros, bits, pack_num, num_groups, out_features, scales
                )
                # Groups run along in_features; scales broadcast over out
                for g in range(num_groups):
                    s, e = g * group_size, min((g + 1) * group_size, in_features)
                    weight[s:e, :] = (
                        weight[s:e, :] - zeros[g : g + 1, :]
                    ) * scales[g : g + 1, :]

                return weight.T  # → [out, in]

            else:
                raise ValueError(
                    f"Cannot determine packing layout – "
                    f"qweight {qweight.shape}, scales {scales.shape}"
                )

        except Exception as exc:
            logger.error(f"Packed-weight dequant failed: {exc}")
            raise

    # ── zero-point unpacking helper ─────────────────────────────────────
    @staticmethod
    def _unpack_zeros(
        qzeros: Optional[torch.Tensor],
        bits: int,
        pack_num: int,
        num_groups: int,
        out_features: int,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Unpack quantised zero-point tensor or return a zero matrix."""
        if qzeros is None:
            return torch.zeros_like(scales)

        zeros = torch.zeros(num_groups, out_features, dtype=torch.float32)
        for j in range(qzeros.shape[1]):
            for k in range(pack_num):
                col = j * pack_num + k
                if col < out_features:
                    zeros[:, col] = (
                        (qzeros[:, j] >> (bits * k)) & ((1 << bits) - 1)
                    ).float()
        return zeros

    # ── enumerate linear layers ─────────────────────────────────────────
    @staticmethod
    def get_all_linear_layer_names(
        model, prefix: str = "model.layers"
    ) -> List[str]:
        """Return dotted names of every linear-like layer under *prefix*."""
        names: List[str] = []
        for name, mod in model.named_modules():
            if not name.startswith(prefix):
                continue
            is_linear = (
                isinstance(mod, torch.nn.Linear)
                or hasattr(mod, "qweight")
                or (hasattr(mod, "weight") and hasattr(mod.weight, "quant_state"))
                or (hasattr(mod, "dequantize") and callable(mod.dequantize))  # HQQ
            )
            if is_linear:
                names.append(name)
        return names


# ═══════════════════════════════════════════════════════════════════════════
#  WeightCache – sample, save, reload, compare
# ═══════════════════════════════════════════════════════════════════════════
class WeightCache:
    """Manages sampled weight vectors on disk for offline analysis."""

    def __init__(
        self,
        cache_dir: str,
        sample_size: int = 500_000,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.rng = np.random.RandomState(seed)
        self.extractor = WeightExtractor()

    # ── save ────────────────────────────────────────────────────────────
    def save_layer_weights(
        self,
        model,
        layer_name: str,
        tag: str,
        full: bool = False,
        force: bool = False,
    ) -> Path:
        """
        Extract, (optionally sample), and persist weights for one layer.

        Args:
            model: The loaded model.
            layer_name: Dotted path (e.g. ``model.layers.30.mlp.down_proj``).
            tag: Quantisation identifier (``bf16``, ``gptq``, …).
            full: Save every weight value (large!) or a random sample.
            force: Re-extract even if the cached file already exists.

        Returns:
            Path to the saved ``.npz`` file.
        """
        safe = layer_name.replace(".", "_")
        path = self.cache_dir / tag / f"{safe}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already cached (avoids re-extracting 160+ layers per run)
        if not force and path.exists() and path.stat().st_size > 0:
            return path

        logger.debug(f"Extracting weights: {layer_name} [{tag}]")
        weight = self.extractor.extract_weight(model, layer_name)
        flat = weight.numpy().flatten()

        if full or len(flat) <= self.sample_size:
            np.savez_compressed(
                path,
                values=flat,
                shape=np.array(weight.shape),
                is_sample=np.array(False),
                numel=np.array(len(flat)),
            )
        else:
            idx = self.rng.choice(len(flat), self.sample_size, replace=False)
            idx.sort()
            np.savez_compressed(
                path,
                values=flat[idx],
                indices=idx,
                shape=np.array(weight.shape),
                is_sample=np.array(True),
                numel=np.array(len(flat)),
            )

        logger.debug(
            f"Saved: {path}  ({path.stat().st_size / 1024:.1f} KB)"
        )
        return path

    # ── load ────────────────────────────────────────────────────────────
    def load_layer_weights(
        self, layer_name: str, tag: str
    ) -> Dict[str, Any]:
        """
        Load previously cached weight data.

        Returns:
            dict with keys ``values``, ``shape``, ``is_sample``,
            ``numel``, and optionally ``indices``.
        """
        safe = layer_name.replace(".", "_")
        path = self.cache_dir / tag / f"{safe}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached weights for {layer_name} [{tag}] at {path}"
            )
        data = np.load(path)
        return {
            "values": data["values"],
            "shape": tuple(data["shape"]),
            "is_sample": bool(data["is_sample"]),
            "numel": int(data["numel"]),
            "indices": data["indices"] if "indices" in data else None,
        }

    # ── MSE ─────────────────────────────────────────────────────────────
    def compute_mse(
        self, layer_name: str, tag_a: str, tag_b: str
    ) -> float:
        """
        Compute MSE between two cached weight samples.

        Both must have been saved with the same random seed so that
        sampled indices align.
        """
        a = self.load_layer_weights(layer_name, tag_a)
        b = self.load_layer_weights(layer_name, tag_b)
        va, vb = a["values"], b["values"]

        # Both sampled with matching indices → direct comparison
        if (
            a["is_sample"]
            and b["is_sample"]
            and a.get("indices") is not None
            and b.get("indices") is not None
            and np.array_equal(a["indices"], b["indices"])
        ):
            return float(np.mean((va - vb) ** 2))

        # Both full
        if not a["is_sample"] and not b["is_sample"]:
            n = min(len(va), len(vb))
            return float(np.mean((va[:n] - vb[:n]) ** 2))

        # Mixed: sample from the full vector using the other's indices
        if a["is_sample"] and not b["is_sample"]:
            return float(np.mean((va - vb[a["indices"]]) ** 2))

        if not a["is_sample"] and b["is_sample"]:
            return float(np.mean((va[b["indices"]] - vb) ** 2))

        # Fallback: truncate to common length
        n = min(len(va), len(vb))
        return float(np.mean((va[:n] - vb[:n]) ** 2))

    # ── validation ──────────────────────────────────────────────────────
    def validate_caches(self) -> Dict[str, Any]:
        """
        Validate weight cache integrity and consistency.

        Returns:
            Dict with 'valid' (bool), 'tags' (list of tags found),
            'issues' (list of problems), and 'layer_counts' (dict of tag -> count).
        """
        issues: List[str] = []
        tags: List[str] = []
        layer_counts: Dict[str, int] = {}
        
        # Check if cache directory exists
        if not self.cache_dir.exists():
            issues.append(f"Cache directory not found: {self.cache_dir}")
            return {
                "valid": False,
                "tags": tags,
                "issues": issues,
                "layer_counts": layer_counts,
            }
        
        # Scan for quantizer directories
        tag_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        if not tag_dirs:
            issues.append(f"No quantizer directories found in {self.cache_dir}")
            return {
                "valid": False,
                "tags": tags,
                "issues": issues,
                "layer_counts": layer_counts,
            }
        
        # Validate each tag's cache
        for tag_dir in tag_dirs:
            tag = tag_dir.name
            tags.append(tag)
            
            # Count .npz files
            npz_files = list(tag_dir.glob("*.npz"))
            if not npz_files:
                issues.append(f"No .npz files found in {tag_dir}")
                continue
            
            layer_counts[tag] = len(npz_files)
            
            # Validate first file can be read
            try:
                first_npz = npz_files[0]
                data = np.load(first_npz)
                required_keys = {"values", "shape"}
                if not required_keys.issubset(data.keys()):
                    issues.append(
                        f"Missing keys in {first_npz.name}: "
                        f"expected {required_keys}, got {set(data.keys())}"
                    )
                data.close()
            except Exception as exc:
                issues.append(f"Cannot read {first_npz.name}: {exc}")
        
        # Cross-check: BF16 should exist if any quantized format exists
        if "bf16" not in tags and len(tags) > 1:
            issues.append(
                "BF16 baseline not found, but quantized formats exist. "
                "Run Module 1 first."
            )
        
        # Cross-check: layer counts should be similar across tags
        if layer_counts and len(layer_counts) > 1:
            counts = list(layer_counts.values())
            if len(set(counts)) > 1:
                issues.append(
                    f"Inconsistent layer counts: {layer_counts}. "
                    "All formats should cache the same layers."
                )
        
        return {
            "valid": len(issues) == 0,
            "tags": tags,
            "issues": issues,
            "layer_counts": layer_counts,
        }

    # ── JSON helpers ────────────────────────────────────────────────────
    def save_json(self, data: Any, filename: str) -> Path:
        path = self.cache_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def load_json(self, filename: str) -> Any:
        path = self.cache_dir / filename
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
