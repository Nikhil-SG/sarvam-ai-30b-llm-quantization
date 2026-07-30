#!/usr/bin/env python3
"""
Module 1a — Fisher Information / Hessian-based Sensitivity Analysis.

Computes the diagonal Fisher Information for each expert within the 18 MoE
layers of sarvamai/sarvam-30b to determine which experts are mathematically
"unstable" and require higher precision to maintain the original weight
distribution.

Architecture reference (from modeling_sarvam_moe.py):
    model.layers.0             → Dense layer (first_k_dense_replace=1)
    model.layers.{1-18}        → MoE layers (18 total)
      .mlp.experts.{0-127}     → 128 routed experts (SarvamMoEMLP)
        .gate_proj / .up_proj / .down_proj  → Linear layers
      .mlp.shared_experts      → 1 shared expert
      .mlp.gate                → SarvamMoEGate (sigmoid routing)

Usage:
    python -m src.mxmoe.sensitivity.fisher_info --calib_samples 512

RUN THIS NEXT: After this, run expert_router_stats.py, then importance_map.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.auth import configure_hf_home, resolve_hf_token, resolve_model_path
from src.core.calibration import load_calibration_data
from src.core.device import build_max_memory_map
from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-30b"
NUM_MOE_LAYERS = 18            # layers 1..18 are MoE
FIRST_MOE_LAYER = 1            # layer 0 is dense
NUM_EXPERTS = 128
NUM_SHARED_EXPERTS = 1


def _build_calibration_dataset(
    tokenizer,
    num_samples: int = 512,
    seq_length: int = 2048,
    seed: int = 42,
    dataset_name: str = "dataset/sangraha_verified",
    dataset_config: Optional[str] = None,
    split: str = "train",
    hf_token: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Build calibration tensors using a shared dataset policy.

    Returns a list of input_ids tensors, each of shape (seq_length,).
    """
    logger.info(
        "Building calibration dataset: %d samples, seq_len=%d, dataset=%s",
        num_samples,
        seq_length,
        dataset_name,
    )

    fallback_chain = []
    if dataset_name != "wikitext":
        fallback_chain.append({
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
        })

    if dataset_name == "wikitext" and dataset_config is None:
        dataset_config = "wikitext-2-raw-v1"

    calibration_data = load_calibration_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
        text_column="text",
        streaming=False,
        min_text_length=200,
        fallback_datasets=fallback_chain,
        hf_token=hf_token,
    )

    input_ids_list = [item["input_ids"] for item in calibration_data]
    logger.info("  Tokenized: %d sequences of length %d", len(input_ids_list), seq_length)
    return input_ids_list


class FisherInformationAnalyzer:
    """
    Compute per-expert diagonal Fisher Information scores.

    The Fisher Information measures the expected curvature of the loss landscape
    w.r.t. model parameters. Experts with high Fisher scores are "sensitive" —
    quantizing them aggressively will damage model quality significantly.

    Algorithm:
        1. Forward pass on calibration data with gradient tracking
        2. Compute loss (next-token prediction)
        3. Backward pass to get gradients
        4. Accumulate squared gradients per expert (diagonal Fisher)
        5. Normalize by number of tokens that activated each expert
    """

    def __init__(
        self,
        config=None,
        model_id: str = MODEL_ID,
        output_dir: str = "mxmoe/outputs/module_1_sensitivity/results",
        calib_samples: int = 512,
        seq_length: int = 2048,
        batch_size: int = 1,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calib_samples = calib_samples
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.seed = seed
        self.calib_dataset = "dataset/sangraha_verified"
        self.calib_dataset_config = None
        self.calib_split = "train"

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
                    self.seed = getattr(cal_cfg, "seed", self.seed)
                    self.calib_dataset = getattr(cal_cfg, "dataset", self.calib_dataset)
                    self.calib_dataset_config = getattr(cal_cfg, "dataset_config", self.calib_dataset_config)
                    self.calib_split = getattr(cal_cfg, "split", self.calib_split)
                fisher_cfg = getattr(sens_cfg, "fisher", None)
                if fisher_cfg:
                    self.batch_size = getattr(fisher_cfg, "accumulate_batches", self.batch_size)
            model_cfg = getattr(config, "model", None)
            if model_cfg:
                self.model_id = getattr(model_cfg, "model_id", self.model_id)

        self.config = config

    def run(self) -> Dict[str, Any]:
        """
        Run the full Fisher Information analysis.

        Returns:
            Dict with fisher_scores, layer summaries, and metadata.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 1a: Fisher Information Analysis")
        logger.info("=" * 60)

        t_start = time.time()

        # ── Load model ───────────────────────────────────────────────────
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

        # ── Build calibration data ───────────────────────────────────────
        calib_data = _build_calibration_dataset(
            tokenizer,
            num_samples=self.calib_samples,
            seq_length=self.seq_length,
            seed=self.seed,
            dataset_name=self.calib_dataset,
            dataset_config=self.calib_dataset_config,
            split=self.calib_split,
            hf_token=hf_token,
        )

        # ── Compute Fisher scores ────────────────────────────────────────
        fisher_scores = self._compute_fisher_scores(model, calib_data)

        # ── Save results ─────────────────────────────────────────────────
        results = {
            "model_id": self.model_id,
            "num_calibration_samples": len(calib_data),
            "seq_length": self.seq_length,
            "num_moe_layers": NUM_MOE_LAYERS,
            "num_experts_per_layer": NUM_EXPERTS,
            "fisher_scores": fisher_scores,
            "total_time_sec": round(time.time() - t_start, 2),
        }

        output_path = self.output_dir / "fisher_scores.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Fisher scores saved: {output_path}")

        # ── Cleanup ──────────────────────────────────────────────────────
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _compute_fisher_scores(
        self,
        model: nn.Module,
        calib_data: List[torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute diagonal Fisher Information for each expert.

        For each calibration sample:
            1. Forward pass → compute NLL loss
            2. Backward pass → accumulate grad²
            3. Map gradients to their owning expert

        Returns:
            {layer_idx_str: {expert_idx_str: fisher_score, "shared": score}}
        """
        model.eval()

        # ── Register gradient accumulators for expert parameters ─────────
        # Maps: (layer_idx, expert_idx) → accumulated squared gradient norm
        fisher_accum: Dict[Tuple[int, int], float] = defaultdict(float)
        fisher_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        # For shared experts: key = (layer_idx, -1)

        # Build parameter → (layer, expert) mapping
        param_to_expert: Dict[int, Tuple[int, int]] = {}
        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            layer = model.model.layers[layer_idx]
            mlp = layer.mlp

            # Routed experts
            for expert_idx in range(NUM_EXPERTS):
                expert = mlp.experts[expert_idx]
                for param in expert.parameters():
                    param_to_expert[id(param)] = (layer_idx, expert_idx)

            # Shared expert
            if hasattr(mlp, "shared_experts") and mlp.shared_experts is not None:
                for param in mlp.shared_experts.parameters():
                    param_to_expert[id(param)] = (layer_idx, -1)  # -1 = shared

        logger.info(f"Tracking {len(param_to_expert)} expert parameters across {NUM_MOE_LAYERS} MoE layers")

        # ── Process calibration samples ──────────────────────────────────
        num_processed = 0
        for sample_idx, input_ids in enumerate(tqdm(calib_data, desc="Fisher computation")):
            input_ids = input_ids.unsqueeze(0)  # (1, seq_len)

            # Determine device from model
            first_param = next(model.parameters())
            input_ids = input_ids.to(first_param.device)

            # Forward pass with gradient tracking
            model.zero_grad(set_to_none=True)
            try:
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,  # causal LM: shift is handled internally
                )
                loss = outputs.loss
                if loss is None:
                    continue

                # Backward pass
                loss.backward()

                # Accumulate squared gradients for expert parameters
                for param in model.parameters():
                    pid = id(param)
                    if pid in param_to_expert and param.grad is not None:
                        layer_idx, expert_idx = param_to_expert[pid]
                        grad_sq_norm = param.grad.float().pow(2).sum().item()
                        fisher_accum[(layer_idx, expert_idx)] += grad_sq_norm
                        fisher_counts[(layer_idx, expert_idx)] += 1

                num_processed += 1

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at sample {sample_idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Periodic GPU cleanup
            if (sample_idx + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        logger.info(f"Processed {num_processed}/{len(calib_data)} calibration samples")

        # ── Normalize and structure results ──────────────────────────────
        fisher_scores: Dict[str, Dict[str, float]] = {}

        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            layer_scores: Dict[str, float] = {}

            for expert_idx in range(NUM_EXPERTS):
                key = (layer_idx, expert_idx)
                count = fisher_counts.get(key, 1)
                score = fisher_accum.get(key, 0.0) / max(count, 1)
                layer_scores[str(expert_idx)] = score

            # Shared expert
            shared_key = (layer_idx, -1)
            shared_count = fisher_counts.get(shared_key, 1)
            shared_score = fisher_accum.get(shared_key, 0.0) / max(shared_count, 1)
            layer_scores["shared"] = shared_score

            fisher_scores[str(layer_idx)] = layer_scores

        return fisher_scores


def main():
    parser = argparse.ArgumentParser(
        description="Module 1a: Fisher Information Analysis for sarvam-30b MoE experts"
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
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for Fisher computation (default: 1)",
    )
    args = parser.parse_args()

    analyzer = FisherInformationAnalyzer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        calib_samples=args.calib_samples,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
    )
    results = analyzer.run()

    logger.info(f"Fisher analysis complete. "
                f"Processed {results.get('num_calibration_samples', 0)} samples "
                f"in {results.get('total_time_sec', 0):.1f}s")


if __name__ == "__main__":
    main()
