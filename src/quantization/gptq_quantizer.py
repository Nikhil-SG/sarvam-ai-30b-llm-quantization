"""
GPTQ Quantisation (4-bit post-training quantisation via auto-gptq).

Strategy:
  1. If a pre-quantised model ID is set in config → load directly.
  2. If a locally saved checkpoint exists from a prior run → reload it.
  3. Otherwise, quantise from scratch using calibration data.

GPTQ uses layer-wise Hessian-based optimal quantisation with grouping
(Frantar et al., 2023) for near-lossless 4-bit compression.

Requirements:
    pip install auto-gptq>=0.7.1
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import shutil
import sys
import types
import torch
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# ---------------------------------------------------------------------------
# Fix: optimum.gptq.quantizer references several symbols from auto_gptq:
#   - QuantizeConfig (or BaseQuantizeConfig)
#   - FORMAT (enum from auto_gptq.quantization)
#   - exllama_set_max_input_length (function)
#
# Newer auto-gptq versions removed/renamed these, so optimum's try/except
# imports silently fail, leaving the names undefined.  We create stubs for
# any missing symbols and inject them into:
#   1. auto_gptq (top-level)
#   2. optimum.gptq.quantizer (where the errors surface)
# ---------------------------------------------------------------------------
def _patch_optimum_gptq():
    """Ensure all symbols optimum.gptq.quantizer needs are available."""

    # ── Locate or create QuantizeConfig ──────────────────────────────────
    _qc = None
    try:
        import auto_gptq
    except ImportError:
        auto_gptq = None

    if auto_gptq is not None:
        for attr in ('QuantizeConfig', 'BaseQuantizeConfig'):
            _qc = getattr(auto_gptq, attr, None)
            if _qc is not None:
                break
        if _qc is None:
            for submod_name in ('auto_gptq.quantization',
                                'auto_gptq.quantize_config',
                                'auto_gptq.utils'):
                try:
                    _m = importlib.import_module(submod_name)
                    for attr in ('QuantizeConfig', 'BaseQuantizeConfig'):
                        _qc = getattr(_m, attr, None)
                        if _qc is not None:
                            break
                except (ImportError, ModuleNotFoundError):
                    continue
                if _qc is not None:
                    break

    if _qc is None:
        @dataclass
        class QuantizeConfig:
            bits: int = 4
            group_size: int = 128
            damp_percent: float = 0.01
            desc_act: bool = True
            static_groups: bool = False
            sym: bool = False
            true_sequential: bool = True
            model_name_or_path: Optional[str] = None
            model_file_base_name: Optional[str] = None
            quant_method: str = "gptq"
            format: str = "gptq"
            act_group_aware: bool = False

            def __init__(self, **kwargs):
                # Accept all known fields and silently ignore unknown ones
                for name, fld in self.__dataclass_fields__.items():
                    setattr(self, name, kwargs.get(name, fld.default))
        _qc = QuantizeConfig
    else:
        # Wrap real QuantizeConfig.__init__ to absorb unknown kwargs from
        # newer transformers/optimum versions (e.g. act_group_aware).
        _real_qc = _qc
        _orig_init = _real_qc.__init__

        import functools as _ft
        @_ft.wraps(_orig_init)
        def _safe_init(self, *args, **kwargs):
            import inspect
            sig = inspect.signature(_orig_init)
            valid = set(sig.parameters.keys()) - {'self'}
            # If the original accepts **kwargs, pass everything
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_kw:
                filtered = kwargs
            else:
                filtered = {k: v for k, v in kwargs.items() if k in valid}
            return _orig_init(self, *args, **filtered)

        _real_qc.__init__ = _safe_init

    # ── Locate or create FORMAT enum ────────────────────────────────────
    _fmt = None
    if auto_gptq is not None:
        _fmt = getattr(auto_gptq, 'FORMAT', None)
        if _fmt is None:
            for submod_name in ('auto_gptq.quantization',
                                'auto_gptq.utils'):
                try:
                    _m = importlib.import_module(submod_name)
                    _fmt = getattr(_m, 'FORMAT', None)
                except (ImportError, ModuleNotFoundError):
                    pass
                if _fmt is not None:
                    break

    if _fmt is None:
        class FORMAT(str, Enum):
            GPTQ = "gptq"
            GPTQ_V2 = "gptq_v2"
            MARLIN = "marlin"
            BITBLAS = "bitblas"
        _fmt = FORMAT

    # ── Locate or create METHOD enum ────────────────────────────────────
    # Newer optimum versions reference METHOD (quant_method enum) in
    # GPTQQuantizer.__init__.  If auto_gptq or optimum doesn't define it,
    # we must create a stub.
    _method = None
    if auto_gptq is not None:
        _method = getattr(auto_gptq, 'METHOD', None)
        if _method is None:
            for submod_name in ('auto_gptq.quantization',
                                'auto_gptq.utils'):
                try:
                    _m = importlib.import_module(submod_name)
                    _method = getattr(_m, 'METHOD', None)
                except (ImportError, ModuleNotFoundError):
                    pass
                if _method is not None:
                    break

    if _method is None:
        class METHOD(str, Enum):
            GPTQ = "gptq"
            AWQ = "awq"
        _method = METHOD

    # ── Locate or create exllama_set_max_input_length ───────────────────
    _exl = None
    if auto_gptq is not None:
        _exl = getattr(auto_gptq, 'exllama_set_max_input_length', None)
    if _exl is None:
        def _exl(model, max_input_length=None):
            return model

    # ── Inject into auto_gptq ───────────────────────────────────────────
    if auto_gptq is not None:
        for name, obj in [('QuantizeConfig', _qc),
                          ('FORMAT', _fmt),
                          ('METHOD', _method),
                          ('exllama_set_max_input_length', _exl)]:
            if not hasattr(auto_gptq, name):
                setattr(auto_gptq, name, obj)

    # ── Ensure auto_gptq.quantization sub-module has FORMAT ─────────────
    # optimum does 'from auto_gptq.quantization import FORMAT'; if the
    # sub-module is missing or empty we must create / populate it BEFORE
    # optimum.gptq.quantizer is first imported.
    if auto_gptq is not None:
        try:
            _aq_q = importlib.import_module('auto_gptq.quantization')
        except (ImportError, ModuleNotFoundError):
            _aq_q = types.ModuleType('auto_gptq.quantization')
            _aq_q.__package__ = 'auto_gptq'
            sys.modules['auto_gptq.quantization'] = _aq_q
        for _sym_name, _sym_obj in [('FORMAT', _fmt),
                                     ('QuantizeConfig', _qc),
                                     ('METHOD', _method)]:
            if not hasattr(_aq_q, _sym_name):
                setattr(_aq_q, _sym_name, _sym_obj)

    # ── Inject into optimum.gptq.quantizer ──────────────────────────────
    try:
        _opt = importlib.import_module('optimum.gptq.quantizer')
        for name, obj in [('QuantizeConfig', _qc),
                          ('FORMAT', _fmt),
                          ('METHOD', _method),
                          ('exllama_set_max_input_length', _exl)]:
            if not hasattr(_opt, name) or getattr(_opt, name) is None:
                setattr(_opt, name, obj)
    except (ImportError, ModuleNotFoundError):
        pass

    # ── Guard auto_gptq fasterquant when nsamples==0 ───────────────────
    # In MoE models some experts may receive zero calibration batches in a
    # block; auto_gptq's logger divides by self.nsamples and crashes.
    try:
        _gptq_mod = importlib.import_module('auto_gptq.quantization.gptq')
        _gptq_cls = getattr(_gptq_mod, 'GPTQ', None)
        if _gptq_cls is not None and not hasattr(_gptq_cls, '_sarvam_nsamples_guard_patch'):
            _orig_fasterquant = _gptq_cls.fasterquant
            _safe_logger = logging.getLogger(__name__)

            def _safe_fasterquant(self, *args, **kwargs):
                if getattr(self, 'nsamples', 0) == 0:
                    _safe_logger.warning(
                        '[gptq] auto_gptq received no calibration batches for this layer; '
                        'forcing nsamples=1 to avoid ZeroDivisionError and continue quantization.'
                    )
                    self.nsamples = 1
                return _orig_fasterquant(self, *args, **kwargs)

            _gptq_cls.fasterquant = _safe_fasterquant
            _gptq_cls._sarvam_nsamples_guard_patch = True
    except Exception:
        pass

_patch_optimum_gptq()

from src.core.logger import get_logger
from src.core.calibration import load_calibration_texts
from src.core.device import resolve_primary_cuda_index
from src.quantization.base import BaseQuantizer

logger = get_logger(__name__)


_DEFAULT_MULTILINGUAL_CALIBRATION_TEXTS = [
    "AI systems for public healthcare must balance throughput, latency, and patient safety under multilingual usage patterns.",
    "Sarvam MoE inference is sensitive to router calibration because expert selection changes with domain and language distribution.",
    "Hindi technical support agents often ask for low latency summarization, secure retrieval, and accurate billing explanations.",
    "भारत में डिजिटल हेल्थ प्लेटफॉर्म को बहुभाषी प्रश्नों के लिए विश्वसनीय, कम विलंबता और सुरक्षित मॉडल चाहिए।",
    "यह मॉडल अंग्रेज़ी और हिंदी दोनों इनपुट पर सही विशेषज्ञ चुन सके, इसलिए कैलिब्रेशन डेटा में मिश्रित तकनीकी वाक्य आवश्यक हैं।",
    "दूरसंचार नेटवर्क ऑटोमेशन में लॉग विश्लेषण, अलर्ट वर्गीकरण और रूट-कॉज़ अनुमान के लिए कम मेमोरी वाले एलएलएम उपयोगी हैं।",
    "Financial assistants need precise entity extraction for GST invoices, reconciliation workflows, and multilingual support tickets.",
    "क्लाउड इन्फ्रास्ट्रक्चर टीम को GPU utilization, memory headroom और request batching का लगातार निरीक्षण करना पड़ता है।",
    "Machine translation for Indic enterprise chat must preserve named entities, technical jargon, and escalation intent.",
    "स्वास्थ्य बीमा दावे की जांच में निदान कोड, दवा इतिहास और अस्पताल के दस्तावेज़ों का सटीक सारांश महत्वपूर्ण है।",
    "MoE routing quality drops when calibration covers only English prose and ignores Hindi, code-mixed, and technical prompts.",
    "बैंकिंग चैटबॉट को यूपीआई विवाद, कार्ड ब्लॉकिंग, और केवाईसी अपडेट जैसे मामलों में स्पष्ट निर्देश देने चाहिए।",
    "A robust GPTQ run should avoid quantizing router projections so expert selection remains stable across mixed-language prompts.",
    "सॉफ्टवेयर इंजीनियर production incident के दौरान error budget, rollback plan और observability dashboard पर निर्भर रहते हैं।",
    "Enterprise copilots must answer questions about compliance, deployment pipelines, and customer escalations in regional languages.",
    "भारतीय उपयोगकर्ता अक्सर English और हिंदी को एक ही वाक्य में मिलाकर लिखते हैं, इसलिए कोड-मिश्रित कैलिब्रेशन उपयोगी है।",
]


def _call_with_supported_kwargs(func, *args, **kwargs):
    """Call a function after dropping unsupported keyword arguments."""
    signature = inspect.signature(func)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        filtered = {k: v for k, v in kwargs.items() if v is not None}
    else:
        filtered = {
            k: v for k, v in kwargs.items()
            if v is not None and k in signature.parameters
        }
    return func(*args, **filtered)


def _json_safe_gptq_value(value):
    """Reduce GPTQ config values to forms that config.json can serialize."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")

    if isinstance(value, torch.Tensor):
        return None

    if isinstance(value, dict):
        safe_dict = {}
        for key, item in value.items():
            safe_item = _json_safe_gptq_value(item)
            if safe_item is not None:
                safe_dict[str(key)] = safe_item
        return safe_dict

    if isinstance(value, (list, tuple)):
        safe_items = []
        for item in value:
            safe_item = _json_safe_gptq_value(item)
            if safe_item is not None:
                safe_items.append(safe_item)
        return safe_items

    return str(value)


def _scrub_gptq_config_for_serialization(model) -> None:
    """Remove calibration-only GPTQ config fields that break save_pretrained."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "quantization_config"):
        return

    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return

    calibration_only_fields = {
        "dataset",
        "tokenizer",
        "calibration_dataset",
        "examples",
    }

    if isinstance(quant_config, dict):
        for field in calibration_only_fields:
            if field in quant_config:
                quant_config[field] = None
        for key, value in list(quant_config.items()):
            quant_config[key] = _json_safe_gptq_value(value)
        return

    for field in calibration_only_fields:
        if hasattr(quant_config, field):
            setattr(quant_config, field, None)

    for key, value in list(vars(quant_config).items()):
        setattr(quant_config, key, _json_safe_gptq_value(value))


def _build_gptqmodel_load_kwargs(config) -> dict:
    """Return only kwargs that are safe to forward into gptqmodel model loading."""
    return {
        "trust_remote_code": getattr(config.model, "trust_remote_code", False),
        "dtype": torch.float16,
    }


def _is_sarvam_moe_config(config) -> bool:
    """Return True when the model config matches Sarvam's custom MoE architecture."""
    if config is None:
        return False

    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or []
    return model_type == "sarvam_moe" or "SarvamMoEForCausalLM" in architectures


def _build_sarvam_gptqmodel_module_tree() -> list:
    """Describe Sarvam's mixed dense-first-layer then MoE layer structure for gptqmodel."""
    return [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": (
                "query_key_value:0",
                "query_layernorm:!",
                "key_layernorm:!",
                "dense:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]


def _configure_gptqmodel_for_sarvam_moe(gptq_model) -> None:
    """Override gptqmodel auto-detection with Sarvam's mixed dense/MoE layer layout."""
    config = getattr(gptq_model, "config", None)
    if not _is_sarvam_moe_config(config):
        return

    lifecycle_hooks = None
    try:
        from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

        lifecycle_hooks = GateUpDownMoELifecycleHooks()
    except Exception:
        lifecycle_hooks = getattr(type(gptq_model), "moe_lifecycle_hooks", None)

    overrides = {
        "module_tree": _build_sarvam_gptqmodel_module_tree(),
        "layer_modules_strict": False,
        "dynamic_expert_index": "num_experts",
        "pre_lm_head_norm_module": "model.norm",
    }
    if lifecycle_hooks is not None:
        overrides["moe_lifecycle_hooks"] = lifecycle_hooks

    for target in (type(gptq_model), gptq_model):
        for name, value in overrides.items():
            setattr(target, name, value)

    logger.info(
        "[gptq] Applied SarvamMoE gptqmodel structure override "
        "(mixed dense layer 0 + MoE layers 1-18)"
    )


def _is_router_layer_name(name: str) -> bool:
    """Return True for MoE routing layers, not MLP gate_proj layers."""
    lowered = name.lower()
    return (
        lowered.endswith(".gate")
        or lowered.endswith(".router")
        or ".router." in lowered
    )


def _infer_module_device(module) -> torch.device:
    """Best-effort device lookup that also works for quantized modules."""
    for attr in ("qweight", "weight", "bias"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value.device

    for parameter in module.parameters(recurse=False):
        return parameter.device

    for buffer in module.buffers(recurse=False):
        return buffer.device

    return torch.device("cpu")


class GPTQQuantizer(BaseQuantizer):
    """4-bit GPTQ quantisation via auto-gptq / transformers integration."""

    QUANT_TAG = "gptq"

    def _allow_transformers_fallback(self) -> bool:
        """Whether the slow transformers GPTQ path is allowed as a fallback."""
        gptq_cfg = getattr(self.config.quantization, "gptq", None)
        return bool(getattr(gptq_cfg, "allow_transformers_fallback", False))

    def _normalize_sym_for_backend(self, sym: bool) -> bool:
        """Adjust sym mode for backend compatibility (auto_gptq does not support sym=False)."""
        backend = getattr(self, "_gptq_backend", None)
        if sym is False and backend == "auto_gptq":
            logger.warning(
                "[gptq] Requested sym=False (asymmetric), but current backend is auto_gptq, "
                "which does not support asymmetric mode. Overriding to sym=True for compatibility. "
                "Install gptqmodel to use sym=False."
            )
            return True
        return bool(sym)

    @staticmethod
    def _has_optimum_gptq() -> bool:
        """Return True when the Optimum GPTQ backend is importable."""
        return importlib.util.find_spec("optimum.gptq") is not None

    def _clear_stale_checkpoint_dir(self) -> None:
        """Remove a previously broken GPTQ checkpoint before a fresh quantization."""
        save_dir = self._saved_quantized_dir
        if save_dir.exists():
            logger.warning(
                f"[gptq] Removing stale checkpoint directory before fresh quantization: {save_dir}"
            )
            shutil.rmtree(save_dir, ignore_errors=True)

    def _get_calibration_strings(self, num_samples: int) -> List[str]:
        """Return calibration strings from config datasets or explicit text overrides."""
        gptq_cfg = getattr(self.config.quantization, "gptq", None)
        cal_cfg = getattr(self.config.quantization, "calibration", None)

        if gptq_cfg is not None:
            explicit_texts = getattr(gptq_cfg, "calibration_texts", None)
            if isinstance(explicit_texts, list):
                cleaned = [str(item).strip() for item in explicit_texts if str(item).strip()]
                if cleaned:
                    logger.info("[gptq] Using explicit calibration_texts from config")
                    return cleaned

            dataset_as_list = getattr(gptq_cfg, "dataset", None)
            if isinstance(dataset_as_list, list):
                cleaned = [str(item).strip() for item in dataset_as_list if str(item).strip()]
                if cleaned:
                    logger.info("[gptq] Using list-form gptq.dataset as calibration texts")
                    return cleaned

        dataset_name = None
        if gptq_cfg is not None:
            value = getattr(gptq_cfg, "dataset", None)
            if isinstance(value, str) and value.strip():
                dataset_name = value.strip()
        if not dataset_name and cal_cfg is not None:
            value = getattr(cal_cfg, "dataset", None)
            if isinstance(value, str) and value.strip():
                dataset_name = value.strip()
        if not dataset_name:
            dataset_name = "dataset/sangraha_verified"

        dataset_config = getattr(gptq_cfg, "dataset_config", None) if gptq_cfg else None
        if dataset_config is None and cal_cfg is not None:
            dataset_config = getattr(cal_cfg, "dataset_config", None)
        if dataset_name == "wikitext" and dataset_config is None:
            dataset_config = "wikitext-2-raw-v1"

        split = getattr(gptq_cfg, "split", None) if gptq_cfg else None
        if not split and cal_cfg is not None:
            split = getattr(cal_cfg, "split", None)
        if not split:
            split = "train"

        seed = getattr(gptq_cfg, "seed", None) if gptq_cfg else None
        if seed is None and cal_cfg is not None:
            seed = getattr(cal_cfg, "seed", None)
        if seed is None:
            seed = 42

        fallback_chain: List[Dict[str, Optional[str]]] = []
        if dataset_name != "wikitext":
            fallback_chain.append({
                "name": "wikitext",
                "config": "wikitext-2-raw-v1",
                "split": "train",
            })

        try:
            texts, source_hits = load_calibration_texts(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
                seed=int(seed),
                text_column="text",
                streaming=False,
                min_text_length=32,
                fallback_datasets=fallback_chain,
                hf_token=self.hf_token,
            )
            if source_hits:
                logger.info("[gptq] Calibration text sources: %s", ", ".join(source_hits))
            return texts
        except Exception as exc:
            logger.warning(
                "[gptq] Calibration dataset loading failed (%s). Falling back to bundled multilingual texts.",
                exc,
            )
            return list(_DEFAULT_MULTILINGUAL_CALIBRATION_TEXTS)

    def _prepare_calibration_texts(
        self,
        tokenizer,
        num_samples: int,
        seq_length: int,
    ) -> List[str]:
        """Build calibration text set with graceful token-length fallback."""
        gptq_cfg = getattr(self.config.quantization, "gptq", None)
        base_min_tokens = int(getattr(gptq_cfg, "min_calibration_tokens", 64)) if gptq_cfg else 64
        base_min_tokens = max(1, base_min_tokens)

        thresholds = [
            base_min_tokens,
            max(32, base_min_tokens // 2),
            16,
            8,
            1,
        ]
        thresholds = sorted(set(thresholds), reverse=True)

        seed_texts = self._get_calibration_strings(num_samples=max(num_samples, 16))
        if not seed_texts:
            raise RuntimeError("No GPTQ calibration seed texts available")

        expanded = list(seed_texts)
        required_pool = max(num_samples * 2, len(seed_texts))
        while len(expanded) < required_pool:
            expanded.extend(seed_texts)
        expanded = expanded[:required_pool]

        token_lengths: List[tuple[str, int]] = []
        skipped_short_text = 0

        for text in tqdm(
            expanded,
            desc="GPTQ calibration prep",
            leave=False,
        ):
            normalized = " ".join(str(text).split())
            if len(normalized) < 32:
                skipped_short_text += 1
                continue

            tokens = tokenizer(
                normalized,
                truncation=True,
                max_length=seq_length,
                return_tensors="pt",
            )
            token_len = int(tokens["input_ids"].shape[1])
            token_lengths.append((normalized, token_len))

        if not token_lengths:
            raise RuntimeError(
                "No valid calibration samples after text normalization. "
                "All candidates were shorter than minimum text length."
            )

        for min_tokens in thresholds:
            prepared = [text for text, tok_len in token_lengths if tok_len >= min_tokens]
            if not prepared:
                logger.warning(
                    "[gptq] Calibration filter min_tokens=%d retained zero samples; relaxing threshold",
                    min_tokens,
                )
                continue

            if len(prepared) < num_samples:
                repeats = (num_samples + len(prepared) - 1) // len(prepared)
                prepared = (prepared * repeats)[:num_samples]
            else:
                prepared = prepared[:num_samples]

            logger.info(
                "Using %d calibration samples for GPTQ (min_tokens=%d, pool=%d, short_text_skips=%d)",
                len(prepared),
                min_tokens,
                len(token_lengths),
                skipped_short_text,
            )
            return prepared

        raise RuntimeError(
            "No valid multilingual calibration samples could be prepared for GPTQ "
            "after token-length fallback thresholds"
        )

    def _tokenize_calibration_texts(
        self,
        texts: List[str],
        tokenizer,
        seq_length: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert calibration strings into tokenized GPTQ examples."""
        tokenized: List[Dict[str, torch.Tensor]] = []
        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_length,
                return_tensors="pt",
            )
            tokenized.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            })
        return tokenized

    def _load_model_for_quantization(self) -> AutoModelForCausalLM:
        """Load the BF16 source model, preferring a single A100 before auto sharding."""
        common_kwargs: Dict[str, Any] = {
            "token": self.hf_token,
            "cache_dir": getattr(self.config.model, "cache_dir", None),
            "trust_remote_code": getattr(self.config.model, "trust_remote_code", False),
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
        }

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            configured_primary = getattr(self.config.hardware, "primary_cuda_index", 1)
            primary_cuda = resolve_primary_cuda_index(configured_primary)

            logger.info(
                f"Quantization load strategy: try full BF16 load on cuda:{primary_cuda} first; "
                "fall back to device_map='auto' if memory is insufficient"
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map={"": primary_cuda},
                    **common_kwargs,
                )
                model.eval()
                return model
            except Exception as exc:
                logger.warning(
                    f"Single-GPU BF16 load on cuda:{primary_cuda} failed ({exc}) — falling back to distributed auto placement"
                )
                torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            **common_kwargs,
        )
        model.eval()
        return model

    def _discover_modules_to_not_convert(self, model) -> List[str]:
        """Identify router layers that must remain in high precision."""
        protected: List[str] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and _is_router_layer_name(name):
                protected.append(name)
        return protected

    def _snapshot_router_layers(self, model, router_layers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Capture BF16 router weights so they can be restored after GPTQ packing."""
        module_map = dict(model.named_modules())
        snapshots: Dict[str, Dict[str, Any]] = {}
        for name in router_layers:
            module = module_map.get(name)
            if not isinstance(module, torch.nn.Linear):
                continue
            snapshots[name] = {
                "weight": module.weight.detach().cpu().clone(),
                "bias": None if module.bias is None else module.bias.detach().cpu().clone(),
                "dtype": module.weight.dtype,
            }
        return snapshots

    def _restore_router_layers(self, model, router_snapshots: Dict[str, Dict[str, Any]]) -> None:
        """Replace quantized router layers with the original BF16 modules."""
        if not router_snapshots:
            return

        restored = 0
        for name, state in router_snapshots.items():
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            current = getattr(parent, attr_name, None)
            target_device = _infer_module_device(current) if current is not None else torch.device("cpu")

            replacement = torch.nn.Linear(
                state["weight"].shape[1],
                state["weight"].shape[0],
                bias=state["bias"] is not None,
                dtype=state["dtype"],
            )
            replacement.weight.data.copy_(state["weight"].to(dtype=state["dtype"]))
            if state["bias"] is not None:
                replacement.bias.data.copy_(state["bias"].to(dtype=state["dtype"]))
            replacement = replacement.to(device=target_device, dtype=state["dtype"])
            setattr(parent, attr_name, replacement)
            restored += 1

        if restored:
            logger.info(
                f"[gptq] Restored {restored} router/gate layers to BF16 after GPTQ packing"
            )

    def _reload_saved_checkpoint(self, save_dir: Path, context: str) -> AutoModelForCausalLM:
        """Reload a saved GPTQ checkpoint using the pipeline's standard model loader."""
        logger.info(f"Reloading quantized model for pipeline [{context}]...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(save_dir),
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        return self.model

    def _assert_model_is_gptq_quantized(self, context: str) -> None:
        """Verify the loaded model still exposes GPTQ quantized linear modules."""
        if self.model is None:
            raise RuntimeError(f"[{context}] No model loaded to validate")

        quantized_layers = []
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if hasattr(module, "qweight") or "QuantLinear" in module_type:
                quantized_layers.append(name)

        if not quantized_layers:
            raise RuntimeError(
                f"[{context}] Loaded model does not expose any GPTQ quantized layers"
            )

        logger.info(
            f"[gptq] Validation [{context}]: detected {len(quantized_layers)} GPTQ quantized layers"
        )

    def _try_load_from_quantized_dir(self) -> bool:
        """Load cached GPTQ checkpoint and reject it if it is not truly quantized."""
        if not super()._try_load_from_quantized_dir():
            return False

        try:
            self._assert_model_is_gptq_quantized("cached_checkpoint")
            return True
        except Exception as exc:
            logger.warning(
                f"[gptq] Cached checkpoint validation failed: {exc} — will re-quantize"
            )
            self.model = None
            return False

    def has_saved_checkpoint(self) -> bool:
        pretrained_id = getattr(self.config.pretrained_quantized, "gptq", None)
        return bool(pretrained_id) or super().has_saved_checkpoint()

    def load_saved_checkpoint_only(self) -> bool:
        pretrained_id = getattr(self.config.pretrained_quantized, "gptq", None)
        if pretrained_id:
            self._quantization_method = "pretrained_gptq"
            self._load_pretrained(pretrained_id)
            return True
        return self._try_load_from_quantized_dir()

    def load_model(self) -> AutoModelForCausalLM:
        # ── 0. Check GPTQ backend availability ─────────────────────────
        #   Newer transformers (>=4.52) requires ``gptqmodel`` instead of
        #   ``auto_gptq``.  Detect early and give a clear error.
        _has_gptqmodel = importlib.util.find_spec("gptqmodel") is not None
        _has_auto_gptq = importlib.util.find_spec("auto_gptq") is not None
        if not _has_gptqmodel and not _has_auto_gptq:
            raise ImportError(
                "GPTQ quantization requires either gptqmodel (`pip install gptqmodel`) "
                "or auto_gptq (`pip install auto-gptq`). Neither is installed."
            )

        self._gptq_backend = "gptqmodel" if _has_gptqmodel else "auto_gptq"
        logger.info("[gptq] Backend detected: %s", self._gptq_backend)

        # ── 1. Pre-quantised model from HF Hub ─────────────────────────
        pretrained_id = getattr(
            self.config.pretrained_quantized, "gptq", None
        )
        if pretrained_id:
            self._quantization_method = "pretrained_gptq"
            return self._load_pretrained(pretrained_id)

        # ── 2. Previously saved checkpoint from quantized_models/ ──────
        if self._try_load_from_quantized_dir():
            self._quantization_method = "cached"
            return self.model

        # ── 3. Quantise from scratch ───────────────────────────────────
        return self._quantize_from_scratch()

    # ── load a pre-quantised checkpoint ─────────────────────────────────
    def _load_pretrained(self, model_id: str) -> AutoModelForCausalLM:
        logger.info(f"Loading pre-quantised GPTQ model: {model_id}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        self._assert_model_is_gptq_quantized("pretrained_checkpoint")
        logger.info("Pre-quantised GPTQ model loaded")
        return self.model

    # ── quantise from scratch ───────────────────────────────────────────
    def _quantize_from_scratch(self) -> AutoModelForCausalLM:
        import gc

        gptq_cfg = self.config.quantization.gptq

        bits = getattr(gptq_cfg, "bits", 4)
        group_size = getattr(gptq_cfg, "group_size", 128)
        desc_act = bool(getattr(gptq_cfg, "desc_act", False))
        damp_percent = float(getattr(gptq_cfg, "damp_percent", 0.1))
        sym = self._normalize_sym_for_backend(getattr(gptq_cfg, "sym", False))

        logger.info(
            f"Quantising {self.model_id} with GPTQ "
            f"(bits={bits}, group_size={group_size}, desc_act={desc_act}, damp_percent={damp_percent}, sym={sym})"
        )
        if desc_act:
            logger.warning(
                "GPTQ desc_act=True is enabled in config, but Sarvam MoE is safer with desc_act=False. "
                "Proceeding with the configured value."
            )

        self._clear_stale_checkpoint_dir()

        cal_cfg = self.config.quantization.calibration
        num_samples = getattr(gptq_cfg, "cal_num_samples",
                              getattr(cal_cfg, "num_samples", 128))
        seq_length = getattr(gptq_cfg, "cal_seq_length",
                             getattr(cal_cfg, "seq_length", 2048))
        batch_size = getattr(gptq_cfg, "batch_size", 1)

        logger.info(
            f"GPTQ calibration: {num_samples} samples, "
            f"seq_length={seq_length}, batch_size={batch_size}"
        )

        tokenizer = self.load_tokenizer()
        calibration_texts = self._prepare_calibration_texts(
            tokenizer=tokenizer,
            num_samples=num_samples,
            seq_length=seq_length,
        )
        calibration_data = self._tokenize_calibration_texts(
            calibration_texts,
            tokenizer=tokenizer,
            seq_length=seq_length,
        )

        # ── Detect architecture ─────────────────────────────────────────
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(
                self.model_id,
                token=self.hf_token,
                trust_remote_code=getattr(
                    self.config.model, "trust_remote_code", False
                ),
                cache_dir=getattr(self.config.model, "cache_dir", None),
            )
            logger.info(
                f"Model architecture: {_cfg.architectures} "
                f"(num_layers={getattr(_cfg, 'num_hidden_layers', '?')})"
            )
        except Exception as exc:
            logger.debug(f"Architecture pre-inspection skipped: {exc}")

        if not self._has_optimum_gptq():
            logger.warning(
                "[gptq] Optimum GPTQ backend (optimum.gptq) is not available. "
                "Trying transformers GPTQ backend directly."
            )
            try:
                return self._quantize_via_transformers(
                    bits=bits,
                    group_size=group_size,
                    desc_act=desc_act,
                    sym=sym,
                    calibration_data=calibration_data,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Optimum GPTQ backend is missing (no module named 'optimum.gptq') and "
                    "transformers GPTQ fallback also failed. Install Optimum in the research env "
                    "(`pip install \"optimum>=1.23,<2\"` or reinstall with `pip install -e \".[research]\"`). "
                    f"Fallback error: {exc}"
                ) from exc

        try:
            return self._quantize_via_optimum(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                damp_percent=damp_percent,
                sym=sym,
                calibration_texts=calibration_texts,
                tokenizer=tokenizer,
                batch_size=batch_size,
                seq_length=seq_length,
            )
        except Exception as exc:
            if (
                not sym
                and "Asymmetric sym=False quantization is not supported with auto-gptq" in str(exc)
            ):
                logger.warning(
                    "[gptq] Optimum rejected sym=False for auto_gptq backend. Retrying with sym=True."
                )
                try:
                    return self._quantize_via_optimum(
                        bits=bits,
                        group_size=group_size,
                        desc_act=desc_act,
                        damp_percent=damp_percent,
                        sym=True,
                        calibration_texts=calibration_texts,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        seq_length=seq_length,
                    )
                except Exception as retry_exc:
                    exc = retry_exc

            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            if not self._allow_transformers_fallback():
                raise RuntimeError(
                    "Optimum GPTQ quantization failed and transformers fallback is disabled. "
                    f"Original error: {exc}"
                ) from exc
            logger.warning(
                f"Optimum GPTQ quantization failed: {exc}. Falling back to transformers GPTQConfig.",
                exc_info=True,
            )

        return self._quantize_via_transformers(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            calibration_data=calibration_data,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    def _quantize_via_optimum(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        damp_percent: float,
        sym: bool,
        calibration_texts: List[str],
        tokenizer,
        batch_size: int = 1,
        seq_length: int = 2048,
    ) -> AutoModelForCausalLM:
        """Quantize with Optimum GPTQ using multilingual calibration and visible tqdm progress."""
        import gc
        from optimum.gptq import GPTQQuantizer as OptimumGPTQQuantizer

        save_dir = self._saved_quantized_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Using Optimum GPTQ quantizer. Progress bars below show block and layer percentage."
        )
        model = self._load_model_for_quantization()

        modules_to_not_convert = self._discover_modules_to_not_convert(model)
        if modules_to_not_convert:
            logger.info(
                "MoE safeguard: keeping router layers in BF16 via restore-after-pack flow: "
                f"{modules_to_not_convert[:5]}"
                f"{' ...' if len(modules_to_not_convert) > 5 else ''}"
            )
        router_snapshots = self._snapshot_router_layers(model, modules_to_not_convert)

        optimum_quantizer = OptimumGPTQQuantizer(
            bits=bits,
            dataset=calibration_texts,
            group_size=group_size,
            damp_percent=damp_percent,
            desc_act=desc_act,
            act_group_aware=not desc_act,
            sym=sym,
            true_sequential=True,
            model_seqlen=seq_length,
            block_name_to_quantize="model.layers",
            batch_size=batch_size,
            pad_token_id=tokenizer.pad_token_id,
            cache_block_outputs=True,
            modules_in_block_to_quantize=None,
            format="gptq",
        )

        quantized_model = optimum_quantizer.quantize_model(model, tokenizer)
        self._restore_router_layers(quantized_model, router_snapshots)
        _scrub_gptq_config_for_serialization(quantized_model)

        logger.info(f"Saving quantized model to {save_dir}")
        optimum_quantizer.save(
            quantized_model,
            str(save_dir),
            max_shard_size="50GB",
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))
        self._write_model_card(save_dir)

        del quantized_model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        self._reload_saved_checkpoint(save_dir, context="optimum")
        self._quantization_method = "optimum_gptq_from_scratch"
        self._save_method = "optimum_save"
        self._assert_model_is_gptq_quantized("optimum_reload")
        self._log_quantized_layers()

        logger.info("GPTQ 4-bit model ready (via optimum)")
        return self.model

    # ── transformers GPTQConfig fallback ────────────────────────────────
    def _quantize_via_transformers(
        self, bits, group_size, desc_act, sym,
        calibration_data, tokenizer, batch_size=1,
    ) -> AutoModelForCausalLM:
        """Fallback: quantize via transformers GPTQConfig backend."""
        logger.info("Using transformers GPTQConfig backend")

        quantization_config = GPTQConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            dataset=calibration_data,
            tokenizer=tokenizer,
        )

        logger.info(
            "Loading model with GPTQ quantization (this takes a while)..."
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        _scrub_gptq_config_for_serialization(self.model)
        self._quantization_method = "transformers_gptq_from_scratch"
        self._assert_model_is_gptq_quantized("transformers_fallback")

        self._log_quantized_layers()

        logger.info("GPTQ 4-bit model ready (via transformers)")
        return self.model

    def save_quantized_model(self) -> Optional[Path]:
        if self._save_method == "optimum_save" and self._saved_quantized_dir.exists():
            return self._saved_quantized_dir
        if self.model is not None:
            _scrub_gptq_config_for_serialization(self.model)
        return super().save_quantized_model()

    # ── helper: log quantized layer summary ────────────────────────────
    def _log_quantized_layers(self):
        """Log summary of quantized linear / expert layers."""
        try:
            linear_layers = []
            expert_layers = []
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                if "Linear" in module_type or "QuantLinear" in module_type:
                    linear_layers.append(name)
                    if "expert" in name.lower():
                        expert_layers.append(name)
            logger.info(
                f"GPTQ quantized {len(linear_layers)} linear layers "
                f"({len(expert_layers)} expert layers detected)"
            )
            if expert_layers:
                logger.info(
                    f"  MoE expert layer sample: {expert_layers[0]}"
                )
            elif linear_layers:
                logger.info(
                    f"  Linear layer sample: {linear_layers[0]}"
                )
        except Exception as exc:
            logger.debug(f"Layer inspection skipped: {exc}")
