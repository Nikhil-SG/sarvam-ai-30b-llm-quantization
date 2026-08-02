"""
FP8 quantisation via optimum-quanto.

Implements a float8 quantizer that preserves the existing quantizer contract:
  - loads or restores a saved checkpoint from quantized_models/fp8_quantized
  - measures memory and caches weights through BaseQuantizer
  - saves a reusable checkpoint for Modules 4 and 5

Design notes:
  - We use optimum-quanto because the repo already supports quanto/float8
    weight extraction in src/core/weight_io.py.
  - Router layers (mlp.gate) and lm_head are excluded from quantization to
    preserve MoE routing stability.
    - We freeze FP8 weights into a saveable checkpoint, while FP8 activations are
        still quantized at runtime after a short calibration pass records scales.
  - Reload uses a dedicated quanto-aware path instead of plain
    AutoModelForCausalLM.from_pretrained().
"""

from __future__ import annotations

import gc
import json
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import get_checkpoint_shard_files, load_state_dict
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from src.core.calibration import load_calibration_texts
from src.core.device import build_cuda_priority_order, resolve_primary_cuda_index
from src.core.logger import get_logger
from src.quantization.base import BaseQuantizer, sanitize_generation_inputs

logger = get_logger(__name__)

_DEFAULT_CALIBRATION_TEXTS = [
    "Bharat is building multilingual language technology for research and production.",
    "Large language models require careful calibration before low-precision deployment.",
    "Sarvam MoE mixes dense attention with routed experts for efficient scaling.",
    "Quantization should reduce memory while preserving generation quality.",
    "Hindi, Tamil, Telugu, Kannada, and English data improve multilingual robustness.",
    "Float8 inference offers a pragmatic middle ground between BF16 and aggressive 4-bit formats.",
]


def _resolve_quanto_qtype(name: Optional[str]):
    if name is None:
        return None

    normalized = str(name).strip().lower()
    if normalized in {"none", "null", "false", "off", ""}:
        return None

    from optimum.quanto import qfloat8, qfloat8_e4m3fn, qfloat8_e5m2, qint8

    mapping = {
        "float8": qfloat8,
        "fp8": qfloat8,
        "qfloat8": qfloat8,
        "float8_e4m3fn": qfloat8_e4m3fn,
        "qfloat8_e4m3fn": qfloat8_e4m3fn,
        "e4m3fn": qfloat8_e4m3fn,
        "float8_e5m2": qfloat8_e5m2,
        "qfloat8_e5m2": qfloat8_e5m2,
        "e5m2": qfloat8_e5m2,
        "int8": qint8,
        "qint8": qint8,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported quanto qtype: {name}")
    return mapping[normalized]


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


class FP8Quantizer(BaseQuantizer):
    """Float8 quantization backed by optimum-quanto."""

    QUANT_TAG = "fp8"

    def has_saved_checkpoint(self) -> bool:
        save_dir = self._saved_quantized_dir
        return (
            save_dir.is_dir()
            and (save_dir / "config.json").exists()
            and self._quanto_qmap_path.exists()
        )

    @property
    def _quanto_qmap_path(self) -> Path:
        return self._saved_quantized_dir / "quanto_qmap.json"

    def _get_fp8_config(self):
        return getattr(self.config.quantization, "fp8", None)

    def _quantized_model_source(self) -> str:
        return self._resolve_local_snapshot() or self.model_id

    def _source_load_kwargs(self, cuda_index: Optional[int] = None) -> Dict[str, Any]:
        if torch.cuda.is_available() and cuda_index is not None:
            device_map: Any = {"": int(cuda_index)}
            torch_dtype = torch.bfloat16
        else:
            device_map = "cpu"
            torch_dtype = torch.float32

        return {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "token": self.hf_token,
            "cache_dir": getattr(self.config.model, "cache_dir", None),
            "trust_remote_code": getattr(self.config.model, "trust_remote_code", False),
        }

    def _fp8_retry_cuda_order(self) -> List[int]:
        """
        Retry order for FP8 loads.

        Prefer configured primary CUDA device (default cuda:1), then fallback.
        """
        configured_primary = getattr(self.config.hardware, "primary_cuda_index", 1)
        return build_cuda_priority_order(configured_primary)

    @staticmethod
    def _cleanup_cuda_between_retries() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _model_device(self, model: Any) -> torch.device:
        base = getattr(model, "model", model)
        try:
            return next(base.parameters()).device
        except Exception:
            if torch.cuda.is_available():
                configured_primary = getattr(self.config.hardware, "primary_cuda_index", 1)
                primary = resolve_primary_cuda_index(configured_primary)
                return torch.device(f"cuda:{primary}")
            return torch.device("cpu")

    def _router_exclude_patterns(self) -> List[str]:
        return [
            "lm_head",
            "*.mlp.gate",
            "model.layers.*.mlp.gate",
        ]

    def _load_calibration_texts(self, num_samples: int) -> List[str]:
        fp8_cfg = self._get_fp8_config()
        shared_cfg = getattr(self.config.quantization, "calibration", None)

        dataset_name = (
            getattr(fp8_cfg, "dataset", None)
            or getattr(shared_cfg, "dataset", "dataset/sangraha_verified")
        )
        dataset_config = getattr(fp8_cfg, "dataset_config", None)
        if dataset_config is None and shared_cfg is not None:
            dataset_config = getattr(shared_cfg, "dataset_config", None)
        if dataset_name == "wikitext" and dataset_config is None:
            dataset_config = "wikitext-2-raw-v1"

        split = getattr(fp8_cfg, "split", None) or getattr(shared_cfg, "split", "train")
        seed = getattr(fp8_cfg, "seed", None)
        if seed is None and shared_cfg is not None:
            seed = getattr(shared_cfg, "seed", 42)
        if seed is None:
            seed = 42

        fallback_chain = []
        if dataset_name != "wikitext":
            fallback_chain.append({
                "name": "wikitext",
                "config": "wikitext-2-raw-v1",
                "split": "train",
            })

        texts: List[str] = []
        try:
            texts, source_hits = load_calibration_texts(
                dataset_name,
                dataset_config,
                split,
                num_samples,
                int(seed),
                text_column="text",
                streaming=False,
                min_text_length=50,
                fallback_datasets=fallback_chain,
                hf_token=self.hf_token,
            )
            if source_hits:
                logger.info("[FP8] Calibration text sources: %s", ", ".join(source_hits))
        except Exception as exc:
            logger.warning(
                f"[FP8] Calibration dataset load failed ({exc}). Using built-in fallback text samples."
            )

        if not texts:
            texts = list(_DEFAULT_CALIBRATION_TEXTS)

        while len(texts) < num_samples:
            texts.extend(_DEFAULT_CALIBRATION_TEXTS)

        return texts[:num_samples]

    def _run_activation_calibration(self, qmodel, activations_qtype) -> None:
        if activations_qtype is None:
            return

        from optimum.quanto import Calibration

        fp8_cfg = self._get_fp8_config()
        shared_cfg = getattr(self.config.quantization, "calibration", None)
        num_samples = getattr(fp8_cfg, "cal_num_samples", None)
        if num_samples is None and shared_cfg is not None:
            num_samples = getattr(shared_cfg, "num_samples", 64)
        seq_length = getattr(fp8_cfg, "cal_seq_length", None)
        if seq_length is None and shared_cfg is not None:
            seq_length = getattr(shared_cfg, "seq_length", 1024)
        batch_size = getattr(fp8_cfg, "batch_size", 4)
        streamline = getattr(fp8_cfg, "streamline", True)

        calibration_texts = self._load_calibration_texts(num_samples)
        target_device = self._model_device(qmodel)

        logger.info(
            f"[FP8] Calibrating activations with {len(calibration_texts)} samples "
            f"(batch_size={batch_size}, seq_length={seq_length})"
        )

        qmodel.eval()
        with torch.no_grad(), Calibration(streamline=streamline):
            for start in tqdm(
                range(0, len(calibration_texts), batch_size),
                desc="FP8 calibration",
                unit="batch",
            ):
                batch_texts = calibration_texts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=seq_length,
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                inputs = sanitize_generation_inputs(qmodel, inputs)
                qmodel(**inputs)

    def _load_quanto_checkpoint(self, save_dir: Path, cuda_index: Optional[int] = None):
        from accelerate import init_empty_weights
        from optimum.quanto import QuantizedModelForCausalLM, requantize
        from optimum.quanto.models.shared_dict import ShardedStateDict

        if torch.cuda.is_available() and cuda_index is not None:
            target_device = torch.device(f"cuda:{int(cuda_index)}")
        else:
            target_device = torch.device("cpu")
        qmap_path = save_dir / "quanto_qmap.json"
        if not qmap_path.exists():
            raise ValueError(f"Quanto quantization map missing: {qmap_path}")

        with open(qmap_path, "r", encoding="utf-8") as fh:
            qmap = json.load(fh)

        config = AutoConfig.from_pretrained(
            str(save_dir),
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(self.config.model, "trust_remote_code", False),
        )

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=getattr(self.config.model, "trust_remote_code", False),
            )

        index_path = save_dir / SAFE_WEIGHTS_INDEX_NAME
        if index_path.exists():
            _, shard_metadata = get_checkpoint_shard_files(str(save_dir), str(index_path))
            state_dict = ShardedStateDict(str(save_dir), shard_metadata["weight_map"])
        else:
            weight_path = save_dir / SAFE_WEIGHTS_NAME
            if not weight_path.exists():
                raise ValueError(f"No safetensor weights found in {save_dir}")
            state_dict = load_state_dict(str(weight_path))

        requantize(model, state_dict=state_dict, quantization_map=qmap, device=target_device)
        model.eval()
        return QuantizedModelForCausalLM(model)

    def _try_load_from_quantized_dir(self) -> bool:
        save_dir = self._saved_quantized_dir
        config_file = save_dir / "config.json"

        if not (save_dir.is_dir() and config_file.exists() and self._quanto_qmap_path.exists()):
            logger.debug(f"[FP8] No saved quanto checkpoint at {save_dir} — will quantize")
            return False

        logger.info(f"[FP8] Found saved checkpoint: {save_dir} — loading from disk")
        cuda_order = self._fp8_retry_cuda_order()
        if not cuda_order:
            cuda_order = [None]

        last_exc: Optional[Exception] = None
        for pos, cuda_index in enumerate(cuda_order):
            try:
                self.model = self._load_quanto_checkpoint(save_dir, cuda_index=cuda_index)
                self.model.eval()
                logger.info(
                    "[FP8] Loaded from saved quanto checkpoint"
                    + (f" on cuda:{cuda_index}" if cuda_index is not None else " on cpu")
                )
                return True
            except Exception as exc:
                last_exc = exc
                self.model = None
                if _is_cuda_oom(exc) and cuda_index is not None and cuda_index != cuda_order[-1]:
                    next_idx = cuda_order[pos + 1]
                    logger.warning(
                        f"[FP8] Saved-checkpoint load OOM on cuda:{cuda_index} ({exc}) — retrying on cuda:{next_idx}"
                    )
                    self._cleanup_cuda_between_retries()
                    continue
                break

        logger.warning(
            f"[FP8] Failed to load saved checkpoint ({last_exc}) — will re-quantize"
        )
        return False

    def load_model(self) -> AutoModelForCausalLM:
        if self._try_load_from_quantized_dir():
            self._quantization_method = "cached"
            return self.model

        try:
            from optimum.quanto import QuantizedModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "FP8 quantization requires optimum-quanto. Install it with `pip install optimum-quanto`."
            ) from exc

        fp8_cfg = self._get_fp8_config()
        weights_qtype = _resolve_quanto_qtype(getattr(fp8_cfg, "weights", "float8"))
        activations_qtype = _resolve_quanto_qtype(getattr(fp8_cfg, "activations", "none"))
        model_src = self._quantized_model_source()
        logger.info(
            f"[FP8] Loading source model for quantization from {model_src} "
            f"(weights={getattr(weights_qtype, 'name', weights_qtype)}, "
            f"activations={getattr(activations_qtype, 'name', activations_qtype) if activations_qtype else 'none'})"
        )

        cuda_order = self._fp8_retry_cuda_order()
        if not cuda_order:
            cuda_order = [None]

        base_model = None
        last_exc: Optional[Exception] = None
        for pos, cuda_index in enumerate(cuda_order):
            try:
                load_kwargs = self._source_load_kwargs(cuda_index=cuda_index)
                base_model = AutoModelForCausalLM.from_pretrained(model_src, **load_kwargs)
                if cuda_index is not None:
                    logger.info(f"[FP8] Source model loaded on cuda:{cuda_index}")
                else:
                    logger.info("[FP8] Source model loaded on CPU")
                break
            except Exception as exc:
                last_exc = exc
                if _is_cuda_oom(exc) and cuda_index is not None and cuda_index != cuda_order[-1]:
                    next_idx = cuda_order[pos + 1]
                    logger.warning(
                        f"[FP8] OOM on cuda:{cuda_index} while loading source model ({exc}) — retrying on cuda:{next_idx}"
                    )
                    self._cleanup_cuda_between_retries()
                    continue
                raise

        if base_model is None:
            attempted = ", ".join(
                [f"cuda:{idx}" for idx in cuda_order if idx is not None]
            ) or "cpu"
            msg = (
                f"[FP8] Could not load source model: CUDA OOM on {attempted}"
                if last_exc and _is_cuda_oom(last_exc)
                else f"[FP8] Could not load source model: {last_exc}"
            )
            raise RuntimeError(msg)

        base_model.eval()

        qmodel = QuantizedModelForCausalLM.quantize(
            base_model,
            weights=weights_qtype,
            activations=activations_qtype,
            exclude=self._router_exclude_patterns(),
        )

        self.model = qmodel
        self._run_activation_calibration(qmodel, activations_qtype)
        self._quantization_method = "optimum_quanto_fp8"

        logger.info("[FP8] Quanto FP8 model ready")
        return self.model

    def save_quantized_model(self) -> Optional[Path]:
        if self.model is None:
            logger.warning("[FP8] No model loaded — skip save")
            return None

        save_dir = self._saved_quantized_dir
        if save_dir.is_dir() and self._quanto_qmap_path.exists():
            weight_files = list(save_dir.rglob("*.safetensors"))
            if weight_files:
                logger.info(f"[FP8] Checkpoint already exists at {save_dir} — skipping save")
                self._save_method = "already_saved"
                return save_dir

        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[FP8] Saving quantized model to {save_dir}")

        try:
            self.model.save_pretrained(str(save_dir))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_dir))
            self._write_model_card(save_dir)
            self._save_method = "optimum_quanto"
            logger.info(f"[FP8] Quantized model saved: {save_dir}")
            return save_dir
        except Exception as exc:
            logger.warning(f"[FP8] save_pretrained failed ({exc})")
            self._save_method = "failed"
            return None