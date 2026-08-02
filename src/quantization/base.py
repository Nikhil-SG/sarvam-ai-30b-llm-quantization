"""
Abstract base class shared by every quantisation back-end.

Provides:
  - tokeniser loading
  - weight caching (target layers + all-layer MSE sweep)
  - static / dynamic memory measurement
  - model architecture auto-detection
  - JSON result persistence
  - a ``run()`` pipeline:  load → measure → cache → unload
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import shutil
import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.logger import get_logger
from src.core.memory import (
    cleanup_model,
    get_memory_snapshot,
    get_peak_memory,
    reset_peak_memory,
    track_memory,
)
from src.core.device import build_max_memory_map
from src.core.auth import resolve_hf_token, resolve_model_path
from src.core.weight_io import WeightCache

logger = get_logger(__name__)


def sanitize_generation_inputs(model, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop tokenizer fields that the model does not accept during generation."""
    cleaned = dict(inputs)

    try:
        accepted = set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        accepted = set()

    if "token_type_ids" in cleaned and "token_type_ids" not in accepted:
        cleaned.pop("token_type_ids", None)

    return cleaned


class BaseQuantizer(ABC):
    """
    Every concrete quantiser must set ``QUANT_TAG`` and implement
    ``load_model()``.  Everything else is handled here.
    """

    QUANT_TAG: str = "base"

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.arch_info: Optional[Dict[str, Any]] = None
        self._quantization_method: Optional[str] = None
        self._save_method: Optional[str] = None

        self.model_id: str = resolve_model_path(config)
        self.hf_token: Optional[str] = resolve_hf_token(config)
        _primary_cuda = getattr(getattr(config, "hardware", None), "primary_cuda_index", 1)
        _max_memory_cfg = (
            config.hardware.max_memory._data
            if hasattr(config.hardware, "max_memory")
            else None
        )
        self.max_memory: Dict = build_max_memory_map(
            _max_memory_cfg,
            primary_cuda_index=_primary_cuda,
        )

        # Quantized loading (INT8/NF4/GPTQ) temporarily holds weights in
        # their original dtype before converting, causing transient memory
        # peaks.  Reduce per-GPU max_memory to leave headroom.
        # 4-bit methods (NF4/GPTQ) need much more headroom because
        # device_map estimates layer sizes in 4-bit but loading uses BF16.
        if self.QUANT_TAG != "bf16":
            # 4-bit: device_map over-packs GPUs, need aggressive reduction
            # 8-bit: moderate reduction is enough
            _is_4bit = self.QUANT_TAG in ("nf4", "gptq")
            _reduction = 40 if _is_4bit else 15
            _floor = 30 if _is_4bit else 40
            _reduced: Dict = {}
            for k, v in self.max_memory.items():
                if isinstance(k, int) and isinstance(v, str) and "GiB" in v:
                    gb = float(v.replace("GiB", ""))
                    _reduced[k] = f"{max(gb - _reduction, _floor):.0f}GiB"
                else:
                    _reduced[k] = v
            self.max_memory = _reduced

        self.results_dir = Path(config.output.results_dir)
        self.weights_dir = Path(config.output.weights_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.weight_cache = WeightCache(
            cache_dir=str(self.weights_dir),
            sample_size=config.visualization.weight_sample_size,
        )

    # ── architecture detection ──────────────────────────────────────────
    def detect_architecture(self) -> Dict[str, Any]:
        """Auto-detect model architecture after loading."""
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")
        from src.core.model_utils import detect_architecture
        self.arch_info = detect_architecture(self.model)
        return self.arch_info

    # ── tokeniser ───────────────────────────────────────────────────────
    def load_tokenizer(self) -> AutoTokenizer:
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                trust_remote_code=getattr(
                    self.config.model, "trust_remote_code", False
                ),
                cache_dir=getattr(self.config.model, "cache_dir", None),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    # ── saved quantized model resolution ───────────────────────────────
    @property
    def _saved_quantized_dir(self) -> Path:
        """Path where save_quantized_model() writes this quantizer's checkpoint."""
        base_dir = getattr(
            self.config.output, "quantized_models_dir", "quantized_models"
        )
        return Path(base_dir) / f"{self.QUANT_TAG}_quantized"

    @property
    def _unsafe_checkpoint_marker(self) -> Path:
        """Marker file used to disable cached reuse of unsafe checkpoints."""
        return self._saved_quantized_dir / ".unsafe_checkpoint"

    def _is_bitsandbytes_backend(self) -> bool:
        """Return True for quantizers that rely on bitsandbytes state."""
        return self.QUANT_TAG in {"int8", "nf4"}

    def _clear_unsafe_checkpoint_marker(self, save_dir: Path) -> None:
        """Remove stale unsafe marker after a known-good save path."""
        marker = save_dir / ".unsafe_checkpoint"
        if marker.exists():
            marker.unlink()
            logger.info(f"[{self.QUANT_TAG}] Cleared stale unsafe checkpoint marker")

    def _mark_checkpoint_unsafe(self, save_dir: Path, reason: str) -> None:
        """Mark a checkpoint as unsafe so future cached loads are skipped."""
        marker = save_dir / ".unsafe_checkpoint"
        marker.write_text(reason.rstrip() + "\n", encoding="utf-8")
        logger.error(
            f"[{self.QUANT_TAG}] Marked checkpoint as unsafe for cached reuse: {marker}"
        )

    def has_saved_checkpoint(self) -> bool:
        """Return True when a reusable quantized checkpoint exists on disk."""
        save_dir = self._saved_quantized_dir
        return (
            save_dir.is_dir()
            and (save_dir / "config.json").exists()
            and not self._unsafe_checkpoint_marker.exists()
        )

    def load_saved_checkpoint_only(self) -> bool:
        """Try to load an existing checkpoint without falling back to quantization."""
        return self._try_load_from_quantized_dir()

    def _try_load_from_quantized_dir(self) -> bool:
        """
        If a previously saved quantized checkpoint exists in
        ``quantized_models/<tag>_quantized/``, load it directly and return
        True (skipping re-quantization).  Returns False if the directory
        doesn't exist or loading fails.

        This is a no-op for quantizers that handle their own on-disk
        checkpoint — they call their own loading logic inside
        ``load_model()``.
        """
        save_dir = self._saved_quantized_dir
        config_file = save_dir / "config.json"

        if not (save_dir.is_dir() and config_file.exists()):
            logger.debug(
                f"[{self.QUANT_TAG}] No saved checkpoint at {save_dir} — will quantize"
            )
            return False

        if self._unsafe_checkpoint_marker.exists():
            reason = ""
            try:
                reason = self._unsafe_checkpoint_marker.read_text(
                    encoding="utf-8"
                ).strip()
            except Exception:
                reason = ""
            reason_suffix = f" Reason: {reason}" if reason else ""
            logger.warning(
                f"[{self.QUANT_TAG}] Checkpoint at {save_dir} is marked unsafe for "
                f"cached reuse.{reason_suffix} — will re-quantize"
            )
            return False

        logger.info(
            f"[{self.QUANT_TAG}] Found saved checkpoint: {save_dir} — loading from disk"
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(save_dir),
                device_map=self.config.hardware.device_map,
                max_memory=self.max_memory,
                token=self.hf_token,
                trust_remote_code=getattr(
                    self.config.model, "trust_remote_code", False
                ),
                cache_dir=getattr(self.config.model, "cache_dir", None),
            )
            self.model.eval()
            logger.info(f"[{self.QUANT_TAG}] Loaded from saved checkpoint (no re-quantization)")
            return True
        except Exception as exc:
            logger.warning(
                f"[{self.QUANT_TAG}] Failed to load from {save_dir}: {exc} — will re-quantize"
            )
            self.model = None
            return False

    # ── local snapshot resolution ───────────────────────────────────────
    def _resolve_local_snapshot(self) -> Optional[str]:
        """
        Check if the model is already cached locally (e.g. downloaded by
        an earlier module).  Returns the local snapshot path or ``None``.
        """
        try:
            from huggingface_hub import scan_cache_dir

            cache_dir = getattr(self.config.model, "cache_dir", None)
            hf_home = os.environ.get("HF_HOME")
            scan_dir = cache_dir or (
                os.path.join(hf_home, "hub") if hf_home else None
            )
            if not scan_dir:
                return None

            hub_dir = scan_dir
            if not hub_dir.endswith("hub"):
                hub_dir = os.path.join(scan_dir, "hub")
                if not os.path.isdir(hub_dir):
                    hub_dir = scan_dir

            cache_info = scan_cache_dir(hub_dir)
            model_slug = self.model_id.replace("/", "--")
            for repo in cache_info.repos:
                if model_slug in str(repo.repo_path):
                    for rev in sorted(
                        repo.revisions,
                        key=lambda r: r.last_modified,
                        reverse=True,
                    ):
                        snap = str(rev.snapshot_path)
                        if os.path.isdir(snap):
                            logger.info(f"Found local snapshot: {snap}")
                            return snap
        except Exception as exc:
            logger.debug(f"Local snapshot resolution failed: {exc}")
        return None

    # ── model (subclass must implement) ─────────────────────────────────
    @abstractmethod
    def load_model(self) -> AutoModelForCausalLM:
        """Load (or quantise) the model and store it in ``self.model``."""

    # ── weight caching ──────────────────────────────────────────────────
    def cache_target_weights(self) -> None:
        """Cache weight samples for the user-selected visualisation layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded – call load_model() first.")

        layers = self.config.visualization.target_layers
        logger.info(f"[{self.QUANT_TAG}] Caching {len(layers)} target layers")
        cached, skipped, errors = 0, 0, 0
        for name in layers:
            try:
                safe = name.replace(".", "_")
                path = self.weights_dir / self.QUANT_TAG / f"{safe}.npz"
                if path.exists() and path.stat().st_size > 0:
                    skipped += 1
                    continue
                self.weight_cache.save_layer_weights(
                    self.model, name, self.QUANT_TAG
                )
                cached += 1
                logger.debug(f"  cached: {name}")
            except Exception as exc:
                errors += 1
                logger.warning(f"  skip {name}: {exc}")
        if skipped:
            logger.info(f"  [{self.QUANT_TAG}] {skipped} target layers already cached — skipped")
        if cached:
            logger.info(f"  [{self.QUANT_TAG}] {cached} target layers newly cached")

    def cache_all_layer_weights(self) -> None:
        """Cache weight samples for every layer × projection (MSE heatmap).

        Uses auto-detected architecture to discover layer count and
        projection names, falling back to config values if detection
        hasn't run yet.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded – call load_model() first.")

        # Use auto-detected architecture if available
        if self.arch_info:
            from src.core.model_utils import get_target_layers_for_heatmap
            all_layers = get_target_layers_for_heatmap(self.arch_info)
            total = len(all_layers)
        else:
            # Fallback to config-based approach
            projs = self.config.visualization.mse_heatmap.projections
            n_layers = self.config.visualization.mse_heatmap.num_layers
            all_layers = []
            for idx in range(n_layers):
                for proj in projs:
                    all_layers.append(f"model.layers.{idx}.{proj}")
            total = len(all_layers)

        # Quick pre-check: count how many are already cached
        already = 0
        for layer_name in all_layers:
            safe = layer_name.replace(".", "_")
            path = self.weights_dir / self.QUANT_TAG / f"{safe}.npz"
            if path.exists() and path.stat().st_size > 0:
                already += 1

        if already == total:
            logger.info(
                f"[{self.QUANT_TAG}] All {total} weight caches already exist — skipping"
            )
            return

        done = 0
        skipped = 0
        logger.info(
            f"[{self.QUANT_TAG}] Caching all-layer weights: {total} tensors "
            f"({already} already cached, {total - already} remaining)"
        )
        for name in all_layers:
            try:
                safe = name.replace(".", "_")
                path = self.weights_dir / self.QUANT_TAG / f"{safe}.npz"
                if path.exists() and path.stat().st_size > 0:
                    skipped += 1
                    done += 1
                    continue
                self.weight_cache.save_layer_weights(
                    self.model, name, self.QUANT_TAG
                )
                done += 1
                if (done - skipped) % 20 == 0 and done > skipped:
                    logger.info(f"  [{self.QUANT_TAG}] {done}/{total} cached")
            except Exception as exc:
                logger.warning(f"  skip {name}: {exc}")

        logger.info(f"[{self.QUANT_TAG}] Cached {done}/{total} layers ({skipped} skipped, {done - skipped} new)")

    # ── memory measurement ──────────────────────────────────────────────
    def measure_static_memory(self) -> Dict[str, Any]:
        """Count parameters and bytes of the loaded model."""
        snap = get_memory_snapshot()
        total_params = 0
        total_bytes = 0

        if self.model is not None:
            for p in self.model.parameters():
                total_params += p.numel()
                # Some quantisation backends wrap data in a differently-sized
                # storage internally.  The actual quantized data may live in
                # p._data.  Use _data's element_size when present
                # to report the true on-GPU footprint, not the wrapper size.
                if hasattr(p, "_data") and isinstance(p._data, torch.Tensor):
                    total_bytes += p.numel() * p._data.element_size()
                else:
                    total_bytes += p.numel() * p.element_size()

        result = {
            "quant_type": self.QUANT_TAG,
            "total_parameters": total_params,
            "model_size_gb": round(total_bytes / (1024 ** 3), 3),
            "memory_snapshot": snap.to_dict(),
        }
        logger.info(
            f"[{self.QUANT_TAG}] {total_params / 1e9:.2f}B params, "
            f"{total_bytes / (1024 ** 3):.2f} GB"
        )
        return result

    def measure_dynamic_memory(
        self, max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate tokens and measure peak / KV-cache memory delta."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model + tokenizer must be loaded first.")

        if max_new_tokens is None:
            max_new_tokens = self.config.profiling.max_new_tokens

        reset_peak_memory()
        before = get_memory_snapshot()

        prompt = self.config.profiling.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # With device_map="auto" the model spans multiple GPUs; use the
        # first parameter's device rather than the undefined model.device.
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        inputs = sanitize_generation_inputs(self.model, inputs)

        with torch.no_grad():
            self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        peak = get_peak_memory()
        after = get_memory_snapshot()

        kv_delta: Dict[int, float] = {}
        for gid in after.gpu_allocated_gb:
            kv_delta[gid] = round(
                after.gpu_allocated_gb[gid]
                - before.gpu_allocated_gb.get(gid, 0),
                3,
            )

        result = {
            "quant_type": self.QUANT_TAG,
            "max_new_tokens": max_new_tokens,
            "peak_memory_gb": peak,
            "kv_cache_delta_gb": kv_delta,
            "before": before.to_dict(),
            "after": after.to_dict(),
        }
        logger.info(f"[{self.QUANT_TAG}] Peak mem: {peak}")
        logger.info(f"[{self.QUANT_TAG}] KV delta: {kv_delta}")
        return result

    # ── persistence ─────────────────────────────────────────────────────
    def save_results(self, results: Dict, filename: str) -> Path:
        path = self.results_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info(f"Results saved: {path}")
        return path

    # ── cleanup ─────────────────────────────────────────────────────────────────
    def unload(self) -> None:
        logger.info(f"[{self.QUANT_TAG}] Unloading model")
        model = self.model
        self.model = None          # drop self reference FIRST
        self.tokenizer = None      # free tokenizer too
        cleanup_model(model)       # now gc can actually collect

    # ── save quantized model to local folder ────────────────────────────
    def save_quantized_model(self) -> Optional[Path]:
        """
        Save the quantized model + tokenizer to a dedicated folder under
        ``config.output.quantized_models_dir``.

        If the checkpoint already exists from a previous run, skip re-saving.

        Subclasses that need special serialisation override this.

        Returns the save directory, or None if the model is not saveable.
        """
        if self.model is None:
            logger.warning(f"[{self.QUANT_TAG}] No model loaded — skip save")
            return None

        base_dir = getattr(self.config.output, "quantized_models_dir", "quantized_models")
        save_dir = Path(base_dir) / f"{self.QUANT_TAG}_quantized"

        if save_dir.is_dir() and self._unsafe_checkpoint_marker.exists():
            logger.warning(
                f"[{self.QUANT_TAG}] Existing checkpoint is marked unsafe at {save_dir} — "
                "removing and re-saving"
            )
            try:
                shutil.rmtree(save_dir)
            except Exception as exc:
                logger.warning(
                    f"[{self.QUANT_TAG}] Could not remove unsafe checkpoint dir "
                    f"({exc})"
                )
                return None

        # Skip if checkpoint already exists from a previous run
        weight_exts = {".safetensors", ".bin", ".pt"}
        if save_dir.is_dir() and any(f.suffix in weight_exts for f in save_dir.iterdir()):
            logger.info(
                f"[{self.QUANT_TAG}] Checkpoint already exists at {save_dir} — skipping save"
            )
            self._save_method = "already_saved"
            return save_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{self.QUANT_TAG}] Saving quantized model to {save_dir}")

        # ── Fix MoE model save: patch _tied_weights_keys ─────────────
        # Some MoE models (e.g. sarvam_moe) have _tied_weights_keys as a
        # list or set which confuses save_pretrained's sharding logic
        # (newer transformers calls .keys() on it, causing AttributeError).
        # Setting to None disables tied-weight resolution during save,
        # which is safe for quantised checkpoints.
        _orig_tied = getattr(self.model, "_tied_weights_keys", None)
        if _orig_tied is not None and not isinstance(_orig_tied, dict):
            try:
                self.model._tied_weights_keys = None
            except Exception:
                pass

        # Also patch hf_device_map if it is somehow a list (MoE dispatch bug)
        _orig_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(_orig_device_map, list):
            try:
                self.model.hf_device_map = {str(i): v for i, v in enumerate(_orig_device_map)}
            except Exception:
                pass

        try:
            self.model.save_pretrained(
                str(save_dir),
                safe_serialization=True,
                max_shard_size="50GB",  # Avoid sharding issues with large MoE models
            )
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_dir))
            self._write_model_card(save_dir)
            self._clear_unsafe_checkpoint_marker(save_dir)
            self._save_method = "save_pretrained"
            logger.info(f"[{self.QUANT_TAG}] Quantized model saved: {save_dir}")
            return save_dir
        except Exception as exc:
            logger.warning(
                f"[{self.QUANT_TAG}] save_pretrained (safetensors) failed ({exc}). "
                f"Trying fallback save strategies."
            )

        # ── Fallback 1: save_pretrained without safe_serialization ──────
        try:
            self.model.save_pretrained(
                str(save_dir),
                safe_serialization=False,
                max_shard_size="50GB",
            )
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_dir))
            self._write_model_card(save_dir)
            self._clear_unsafe_checkpoint_marker(save_dir)
            self._save_method = "save_pretrained_bin"
            logger.info(f"[{self.QUANT_TAG}] Quantized model saved (bin format): {save_dir}")
            return save_dir
        except Exception as exc2:
            logger.warning(
                f"[{self.QUANT_TAG}] save_pretrained (bin) also failed ({exc2}). "
                f"Trying cleaned state_dict approach."
            )

        # ── Fallback 2: save_pretrained with explicit cleaned state_dict ─
        # MoE models can have list-typed metadata in their state that
        # confuses save_pretrained's sharding logic.  Extract only tensors
        # and pass them as an explicit state_dict.
        try:
            state_dict = self.model.state_dict()
            # Filter to only torch.Tensor entries (skip lists, ints, etc.)
            clean_state = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    clean_state[k] = v

            # Save config first (includes quantization_config for bnb)
            if hasattr(self.model, "config"):
                self.model.config.save_pretrained(str(save_dir))

            # Try safetensors with contiguous CPU tensors
            try:
                from safetensors.torch import save_file
                # safetensors requires contiguous tensors; also move to CPU
                safe_state = {}
                for k, v in clean_state.items():
                    t = v.detach().cpu()
                    # safetensors doesn't support int4/uint4; save as uint8
                    if t.dtype in (torch.int8, torch.uint8):
                        safe_state[k] = t.contiguous()
                    elif t.is_floating_point() or t.dtype in (
                        torch.int16, torch.int32, torch.int64,
                    ):
                        safe_state[k] = t.contiguous()
                    else:
                        safe_state[k] = t.to(torch.float16).contiguous()
                save_file(safe_state, str(save_dir / "model.safetensors"))
                logger.info(f"[{self.QUANT_TAG}] Saved via safetensors (cleaned state_dict)")
            except Exception as sf_exc:
                logger.debug(f"safetensors save failed ({sf_exc}), using torch.save")
                torch.save(clean_state, str(save_dir / "pytorch_model.bin"))
                logger.info(f"[{self.QUANT_TAG}] Saved via torch.save fallback")

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_dir))
            self._write_model_card(save_dir)

            if self._is_bitsandbytes_backend():
                reason = (
                    f"{self.QUANT_TAG} checkpoint saved via manual_state_dict fallback. "
                    "This path may drop bitsandbytes quantization metadata; "
                    "cached reuse is disabled."
                )
                self._mark_checkpoint_unsafe(save_dir, reason)
                self._save_method = "manual_state_dict_unsafe"
            else:
                self._clear_unsafe_checkpoint_marker(save_dir)
                self._save_method = "manual_state_dict"

            logger.info(f"[{self.QUANT_TAG}] Quantized model saved (manual): {save_dir}")
            return save_dir
        except Exception as exc3:
            logger.warning(
                f"[{self.QUANT_TAG}] All save strategies failed ({exc3})."
            )
            self._save_method = "failed"
            return None

    # ── generate README / model card ────────────────────────────────────
    def _write_model_card(self, save_dir: Path) -> None:
        """Write a README.md model card so users know how to load the model."""
        model_id = getattr(self.config.model, "model_id", "unknown")
        tag = self.QUANT_TAG.upper()

        load_snippets = {
            "int8": (
                "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n\n"
                f'model = AutoModelForCausalLM.from_pretrained("{model_id}-{tag}", device_map="auto")\n'
                f'tokenizer = AutoTokenizer.from_pretrained("{model_id}-{tag}")'
            ),
            "fp8": (
                "from optimum.quanto import QuantizedModelForCausalLM\n"
                "from transformers import AutoTokenizer\n\n"
                f'model = QuantizedModelForCausalLM.from_pretrained("{model_id}-{tag}")\n'
                f'tokenizer = AutoTokenizer.from_pretrained("{model_id}-{tag}")'
            ),
            "nf4": (
                "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n\n"
                'bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")\n'
                f'model = AutoModelForCausalLM.from_pretrained("{model_id}-{tag}", '
                'quantization_config=bnb_config, device_map="auto")\n'
                f'tokenizer = AutoTokenizer.from_pretrained("{model_id}-{tag}")'
            ),
            "gptq": (
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
                f'model = AutoModelForCausalLM.from_pretrained("{model_id}-{tag}", device_map="auto")\n'
                f'tokenizer = AutoTokenizer.from_pretrained("{model_id}-{tag}")'
            ),
        }

        snippet = load_snippets.get(
            self.QUANT_TAG,
            f'model = AutoModelForCausalLM.from_pretrained("{model_id}-{tag}", device_map="auto")',
        )

        quant_details = {
            "int8": "INT8 via bitsandbytes LLM.int8() \u2014 mixed INT8/FP16 decomposition",
            "fp8": "FP8 via optimum-quanto \u2014 float8 weight quantization with optional activation calibration",
            "nf4": "NF4 via bitsandbytes \u2014 4-bit NormalFloat quantization (QLoRA format)",
            "gptq": "GPTQ \u2014 4-bit post-training quantization with Hessian-based optimization",
        }
        desc = quant_details.get(self.QUANT_TAG, f"{tag} quantized model")

        card = f"""---
library_name: transformers
tags:
  - quantization
  - {self.QUANT_TAG}
base_model: {model_id}
---

# {model_id} — {tag} Quantized

{desc}

**Base model:** [{model_id}](https://huggingface.co/{model_id})

## Usage

```python
{snippet}

# Generate
inputs = tokenizer("Hello, ", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quantization Details

| Property | Value |
|----------|-------|
| Base Model | {model_id} |
| Quantization | {tag} |
| Bits | {self._get_bits()} |

## License

Same as the base model: [{model_id}](https://huggingface.co/{model_id})
"""
        readme_path = save_dir / "README.md"
        readme_path.write_text(card, encoding="utf-8")
        logger.debug(f"[{self.QUANT_TAG}] Model card written: {readme_path}")

    def _get_bits(self) -> str:
        """Return the bit-width string for the model card."""
        bits_map = {"int8": "8", "fp8": "8 (Float8)", "nf4": "4 (NormalFloat)", "gptq": "4"}
        return bits_map.get(self.QUANT_TAG, "unknown")

    # ── full pipeline ───────────────────────────────────────────────────
    def run(self, cache_weights: bool = True) -> Dict[str, Any]:
        """
        End-to-end: load → measure → cache → save → unload.

        Returns:
            dict with ``load_time_sec``, ``static_memory``,
            ``dynamic_memory`` (or error), plus any cached-weight metadata.
        """
        results: Dict[str, Any] = {"quant_type": self.QUANT_TAG}

        try:
            logger.info(f"{'='*60}")
            logger.info(f"  {self.QUANT_TAG.upper()} QUANTIZATION")
            logger.info(f"{'='*60}")

            self.load_tokenizer()

            # ── load model ──────────────────────────────────────────────────────
            with track_memory(f"{self.QUANT_TAG}_load"):
                t0 = time.time()
                self.load_model()
                results["load_time_sec"] = round(time.time() - t0, 2)
            
            # Track which loading method was successful
            if self._quantization_method:
                results["quantization_method"] = self._quantization_method
            
            logger.info(
                f"[{self.QUANT_TAG.upper()}] ✓ Loaded in {results['load_time_sec']:.1f}s"
            )

            # ── architecture detection ──────────────────────────────
            try:
                arch = self.detect_architecture()
                results["architecture"] = arch
            except Exception as exc:
                logger.warning(f"[{self.QUANT_TAG.upper()}] Architecture detection failed: {exc}")

            # ── static memory ───────────────────────────────────────────
            results["static_memory"] = self.measure_static_memory()

            # ── dynamic memory ──────────────────────────────────────────
            try:
                results["dynamic_memory"] = self.measure_dynamic_memory()
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"[{self.QUANT_TAG.upper()}] ⚠ OOM during dynamic measurement"
                )
                results["dynamic_memory"] = {"error": "OutOfMemoryError"}
                torch.cuda.empty_cache()

            # ── cache weights ───────────────────────────────────────────
            if cache_weights:
                try:
                    self.cache_target_weights()
                    self.cache_all_layer_weights()
                    logger.info(f"[{self.QUANT_TAG.upper()}] ✓ Weight caching complete")
                except Exception as exc:
                    logger.warning(
                        f"[{self.QUANT_TAG.upper()}] ⚠ Weight caching failed: {exc}"
                    )

            # ── save quantized model locally ────────────────────────────
            save_dir = self.save_quantized_model()
            # Always record save_method (even on failure) so tests can verify
            if self._save_method:
                results["save_method"] = self._save_method
            if save_dir:
                results["saved_model_dir"] = str(save_dir)
                logger.info(
                    f"[{self.QUANT_TAG.upper()}] ✓ Model persisted to {save_dir}"
                )
            else:
                logger.warning(
                    f"[{self.QUANT_TAG.upper()}] ⚠ Model save failed — checkpoint not available"
                )

            self.save_results(results, f"{self.QUANT_TAG}_results.json")
            logger.info(f"[{self.QUANT_TAG.upper()}] ✓ Results saved")

        except Exception as exc:
            logger.error(
                f"[{self.QUANT_TAG.upper()}] ✗ PIPELINE FAILED: {exc}", exc_info=True
            )
            results["error"] = str(exc)
            # Persist the error result so tests read the *actual* failure
            # instead of a stale result file from a previous run.
            try:
                self.save_results(results, f"{self.QUANT_TAG}_results.json")
            except Exception:
                pass  # don't mask the original error
            raise
        finally:
            self.unload()
            logger.info(f"[{self.QUANT_TAG.upper()}] Unloaded")

        return results
