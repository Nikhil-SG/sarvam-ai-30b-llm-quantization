#!/usr/bin/env python3
"""
Module 1b — Expert Routing Statistics.

Profiles expert activation frequency by hooking into the SarvamMoEGate router
during forward passes on calibration data. Identifies "heavy hitter" experts
(frequently activated, handle bulk of tokens) vs "long tail" experts (rarely
activated, candidates for aggressive quantization).

Architecture reference:
    model.layers.{1-18}.mlp.gate  → SarvamMoEGate
        forward() returns (topk_idx, topk_weight, logits)
        topk_idx shape: (batch * seq_len, 6)  — top-6 expert indices

Usage:
    python -m src.mxmoe.sensitivity.expert_router_stats --calib_samples 512

RUN THIS NEXT: After fisher_info.py, run this, then importance_map.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.core.auth import configure_hf_home, resolve_hf_token, resolve_model_path
from src.core.device import build_max_memory_map
from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-30b"
NUM_MOE_LAYERS = 18
FIRST_MOE_LAYER = 1
NUM_EXPERTS = 128
TOP_K = 6


def _build_calibration_dataset(
    tokenizer,
    num_samples: int = 512,
    seq_length: int = 2048,
) -> List[torch.Tensor]:
    """Build tokenized calibration data (shared with fisher_info.py)."""
    from src.mxmoe.sensitivity.fisher_info import _build_calibration_dataset as _build
    return _build(tokenizer, num_samples=num_samples, seq_length=seq_length)


class ExpertRoutingAnalyzer:
    """
    Track which experts are activated per token across calibration data.

    Uses forward hooks on SarvamMoEGate to intercept the top-k expert
    indices without modifying the model's forward pass.
    """

    def __init__(
        self,
        config=None,
        model_id: str = MODEL_ID,
        output_dir: str = "mxmoe/outputs/module_1_sensitivity/results",
        calib_samples: int = 512,
        seq_length: int = 2048,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calib_samples = calib_samples
        self.seq_length = seq_length

        # Override from config if provided
        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sens_cfg = getattr(config, "sensitivity", None)
            if sens_cfg:
                cal_cfg = getattr(sens_cfg, "calibration", None)
                if cal_cfg:
                    self.calib_samples = getattr(cal_cfg, "num_samples", self.calib_samples)
                    self.seq_length = getattr(cal_cfg, "seq_length", self.seq_length)
            model_cfg = getattr(config, "model", None)
            if model_cfg:
                self.model_id = getattr(model_cfg, "model_id", self.model_id)

        self.config = config

    def run(self) -> Dict[str, Any]:
        """
        Profile expert routing across calibration data.

        Returns:
            Dict with routing_frequency, heavy_hitters, long_tail, and stats.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 1b: Expert Routing Statistics")
        logger.info("=" * 60)

        t_start = time.time()

        # ── Load model (inference only, no grad needed) ──────────────────
        if self.config is not None:
            configure_hf_home(self.config)
            model_source = resolve_model_path(self.config)
            hf_token = resolve_hf_token(self.config)
            model_cache_dir = getattr(getattr(self.config, "model", None), "cache_dir", None)
            trust_remote_code = getattr(getattr(self.config, "model", None), "trust_remote_code", True)
            hardware_cfg = getattr(self.config, "hardware", None)
            primary_cuda_index = getattr(hardware_cfg, "primary_cuda_index", 1) if hardware_cfg else 1
            max_memory_cfg = getattr(hardware_cfg, "max_memory", None) if hardware_cfg else None
        else:
            model_source = self.model_id
            hf_token = None
            model_cache_dir = None
            trust_remote_code = True
            primary_cuda_index = 1
            max_memory_cfg = None

        max_memory = build_max_memory_map(
            max_memory_cfg._data if hasattr(max_memory_cfg, "_data") else max_memory_cfg,
            primary_cuda_index=primary_cuda_index,
        )

        logger.info(f"Loading model: {model_source}")
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            token=hf_token,
            cache_dir=model_cache_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            token=hf_token,
            cache_dir=model_cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()

        # ── Build calibration data ───────────────────────────────────────
        calib_data = _build_calibration_dataset(
            tokenizer,
            num_samples=self.calib_samples,
            seq_length=self.seq_length,
        )

        # ── Register hooks on MoE gates ──────────────────────────────────
        routing_counts, hooks = self._register_routing_hooks(model)

        # ── Run calibration data through the model ───────────────────────
        total_tokens = 0
        with torch.no_grad():
            for sample_idx, input_ids in enumerate(tqdm(calib_data, desc="Routing profiling")):
                input_ids = input_ids.unsqueeze(0)
                first_param = next(model.parameters())
                input_ids = input_ids.to(first_param.device)

                try:
                    model(input_ids=input_ids)
                    total_tokens += input_ids.shape[1]
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM at sample {sample_idx}, skipping")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                if (sample_idx + 1) % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        # ── Remove hooks ─────────────────────────────────────────────────
        for hook in hooks:
            hook.remove()

        # ── Analyze routing patterns ─────────────────────────────────────
        results = self._analyze_routing(routing_counts, total_tokens)
        results["model_id"] = self.model_id
        results["num_calibration_samples"] = len(calib_data)
        results["total_tokens_processed"] = total_tokens
        results["total_time_sec"] = round(time.time() - t_start, 2)

        # ── Save results ─────────────────────────────────────────────────
        output_path = self.output_dir / "routing_stats.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Routing statistics saved: {output_path}")

        # ── Cleanup ──────────────────────────────────────────────────────
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _register_routing_hooks(
        self, model: nn.Module
    ) -> Tuple[Dict[int, Dict[int, int]], List]:
        """
        Register forward hooks on each SarvamMoEGate to capture expert selections.

        Returns:
            (routing_counts dict, list of hook handles)
        """
        # routing_counts[layer_idx][expert_idx] = total activation count
        routing_counts: Dict[int, Dict[int, int]] = {
            layer_idx: defaultdict(int)
            for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS)
        }
        hooks = []

        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            layer = model.model.layers[layer_idx]
            gate = layer.mlp.gate  # SarvamMoEGate

            # Closure to capture layer_idx
            def _make_hook(li: int):
                def _hook(module, input, output):
                    # SarvamMoEGate.forward returns (topk_idx, topk_weight, logits)
                    topk_idx = output[0]  # shape: (num_tokens, top_k)
                    # Count each expert activation
                    expert_indices = topk_idx.detach().cpu().flatten().tolist()
                    for eid in expert_indices:
                        routing_counts[li][int(eid)] += 1
                return _hook

            hook_handle = gate.register_forward_hook(_make_hook(layer_idx))
            hooks.append(hook_handle)

        logger.info(f"Registered {len(hooks)} routing hooks on MoE gates")
        return routing_counts, hooks

    def _analyze_routing(
        self,
        routing_counts: Dict[int, Dict[int, int]],
        total_tokens: int,
    ) -> Dict[str, Any]:
        """
        Analyze routing patterns to identify heavy hitters and long tail.
        """
        results: Dict[str, Any] = {
            "routing_frequency": {},
            "heavy_hitters": {},
            "long_tail": {},
            "layer_summaries": {},
        }

        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            counts = routing_counts[layer_idx]

            # Fill in zeros for never-activated experts
            freq: Dict[str, int] = {}
            for eid in range(NUM_EXPERTS):
                freq[str(eid)] = counts.get(eid, 0)

            total_activations = sum(freq.values())
            results["routing_frequency"][str(layer_idx)] = freq

            # ── Identify heavy hitters and long tail ─────────────────────
            sorted_experts = sorted(
                freq.items(), key=lambda x: x[1], reverse=True
            )

            # Heavy hitters: top 20% of experts by activation count
            top_20_pct = max(1, NUM_EXPERTS // 5)
            heavy = [
                {
                    "expert_id": int(eid),
                    "count": cnt,
                    "pct": round(cnt / max(total_activations, 1) * 100, 2),
                }
                for eid, cnt in sorted_experts[:top_20_pct]
            ]

            # Long tail: bottom 30% with less than 0.5% of activations each
            threshold = total_activations * 0.005
            tail = [
                {
                    "expert_id": int(eid),
                    "count": cnt,
                    "pct": round(cnt / max(total_activations, 1) * 100, 2),
                }
                for eid, cnt in sorted_experts
                if cnt < threshold
            ]

            results["heavy_hitters"][str(layer_idx)] = heavy
            results["long_tail"][str(layer_idx)] = tail

            # ── Layer summary statistics ─────────────────────────────────
            counts_array = np.array([v for v in freq.values()], dtype=np.float64)
            results["layer_summaries"][str(layer_idx)] = {
                "total_activations": total_activations,
                "mean_per_expert": round(float(counts_array.mean()), 2),
                "std_per_expert": round(float(counts_array.std()), 2),
                "min_activations": int(counts_array.min()),
                "max_activations": int(counts_array.max()),
                "num_heavy_hitters": len(heavy),
                "num_long_tail": len(tail),
                "gini_coefficient": round(float(self._gini(counts_array)), 4),
            }

        return results

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        """Compute Gini coefficient — measures routing imbalance (0=uniform, 1=all-to-one)."""
        if len(arr) == 0 or arr.sum() == 0:
            return 0.0
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_arr) - (n + 1) * np.sum(sorted_arr)) / (n * np.sum(sorted_arr)))


def main():
    parser = argparse.ArgumentParser(
        description="Module 1b: Expert Routing Statistics for sarvam-30b"
    )
    parser.add_argument(
        "--model_id", type=str, default=MODEL_ID,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--calib_samples", type=int, default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--seq_length", type=int, default=2048,
        help="Sequence length for calibration (default: 2048)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="mxmoe/outputs/module_1_sensitivity/results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    analyzer = ExpertRoutingAnalyzer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        calib_samples=args.calib_samples,
        seq_length=args.seq_length,
    )
    results = analyzer.run()

    logger.info(f"Routing analysis complete in {results.get('total_time_sec', 0):.1f}s")
    for li, summary in results.get("layer_summaries", {}).items():
        logger.info(
            f"  Layer {li}: gini={summary['gini_coefficient']:.3f}, "
            f"heavy_hitters={summary['num_heavy_hitters']}, "
            f"long_tail={summary['num_long_tail']}"
        )


if __name__ == "__main__":
    main()
