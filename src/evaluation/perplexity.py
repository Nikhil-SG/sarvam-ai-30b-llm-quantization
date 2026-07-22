"""
Module 5a – Perplexity Evaluation (WikiText-2).

Standard sliding-window perplexity using ``model(input_ids, labels=…)``.
"""

from __future__ import annotations

import json
import math
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm import tqdm

from src.core.logger import get_logger

logger = get_logger(__name__)


class PerplexityEvaluator:
    """Compute language-model perplexity on WikiText-2."""

    def __init__(self, config):
        self.config = config
        ppl_cfg = config.evaluation.perplexity
        self.dataset_name = ppl_cfg.dataset
        self.dataset_config = ppl_cfg.dataset_config
        self.split = ppl_cfg.split
        self.max_length = ppl_cfg.max_length
        self.stride = ppl_cfg.stride
        self.max_samples = getattr(ppl_cfg, "max_samples", None)

        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

        # ── Pre-load and cache the eval dataset (load once, reuse for all models) ──
        self._cached_text: Optional[str] = None

    def _get_eval_text(self) -> str:
        """Load and cache the evaluation text. Downloads once, reuses for all models."""
        if self._cached_text is not None:
            logger.info(f"  Using cached {self.dataset_name} text (already loaded)")
            return self._cached_text

        from datasets import load_dataset

        logger.info(f"  Loading {self.dataset_name} (will cache for subsequent models)")
        ds = load_dataset(
            self.dataset_name, self.dataset_config, split=self.split
        )
        self._cached_text = "\n\n".join(ds["text"])
        return self._cached_text

    # ── compute for one model ───────────────────────────────────────────
    def evaluate(
        self, model, tokenizer, tag: str
    ) -> Dict[str, Any]:
        """
        Compute perplexity with a strided sliding window.

        Returns:
            Dict with ``perplexity``, ``avg_nll``, ``num_windows``.
        """
        logger.info(f"Computing perplexity [{tag}] on {self.dataset_name}")

        text = self._get_eval_text()
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        logger.info(f"  Total tokens in eval set: {seq_len:,}")

        nlls = []
        nan_fallback_count = 0
        prev_end = 0

        # With device_map="auto" the model spans multiple GPUs;
        # model.device raises AttributeError.  Use the first parameter's device.
        first_device = next(model.parameters()).device

        # Determine autocast device type and dtype for mixed-precision safety
        _autocast_device = "cuda" if first_device.type == "cuda" else "cpu"
        _autocast_dtype = torch.bfloat16

        for begin in tqdm(
            range(0, seq_len, self.stride),
            desc=f"Perplexity [{tag}]",
            leave=False,
        ):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end
            input_ids = encodings.input_ids[:, begin:end].to(first_device)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad(), torch.amp.autocast(_autocast_device, dtype=_autocast_dtype):
                outputs = model(input_ids, labels=target_ids)
                loss = outputs.loss

                # ── NaN-safe fallback ────────────────────────────────────
                # Some quantized models with broken packing
                # return NaN loss.  Fall back to manual cross-entropy from
                # logits so we always get a finite (if possibly very high)
                # perplexity value.
                if loss is None or not torch.isfinite(loss):
                    nan_fallback_count += 1
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()
                        # Replace any NaN/Inf in logits before softmax
                        shift_logits = torch.nan_to_num(
                            shift_logits, nan=0.0, posinf=1e4, neginf=-1e4
                        )
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                    else:
                        # No logits available — skip this window entirely
                        logger.debug(
                            f"  [{tag}] Window {len(nlls)}: loss=NaN and no logits — skipping"
                        )
                        continue

                nlls.append(loss.detach().cpu())

            prev_end = end
            if end == seq_len:
                break

            if self.max_samples and len(nlls) >= self.max_samples:
                logger.info(
                    f"  Reached max_samples={self.max_samples}, stopping early"
                )
                break

        if nan_fallback_count > 0:
            logger.warning(
                f"  [{tag}] {nan_fallback_count}/{len(nlls) + nan_fallback_count} windows "
                f"had NaN/Inf loss — used manual cross-entropy fallback"
            )

        if not nlls:
            logger.warning(f"  [{tag}] All windows produced NaN — cannot compute perplexity")
            avg_nll = float("nan")
            ppl = float("nan")
        else:
            avg_nll = torch.stack(nlls).mean().item()
            ppl = float(np.exp(avg_nll))

        result = {
            "tag": tag,
            "perplexity": round(ppl, 4),
            "avg_nll": round(avg_nll, 6),
            "num_windows": len(nlls),
            "dataset": self.dataset_name,
            "max_length": self.max_length,
            "stride": self.stride,
        }

        logger.info(f"  [{tag}] Perplexity = {ppl:.4f}  (NLL = {avg_nll:.6f})")
        return result

    # ── compare all tags ────────────────────────────────────────────────
    def plot_comparison(self, ppl_data: Dict[str, Dict]) -> Path:
        """
        Bar chart of perplexity across quantisation formats.

        Args:
            ppl_data: Mapping tag → result dict.
        """
        tag_colors = {
            "bf16": "#2196F3", "gptq": "#4CAF50", "int8": "#E91E63",
        }

        tags, ppls = [], []
        for t, d in ppl_data.items():
            p = d.get("perplexity", None)
            if p is not None and isinstance(p, (int, float)) and not (isinstance(p, float) and math.isnan(p)):
                tags.append(t.upper())
                ppls.append(p)

        if not tags:
            logger.warning("No perplexity data to plot")
            return self.plots_dir / "perplexity_comparison.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        colours = [tag_colors.get(t.lower(), "#888") for t in tags]
        bars = ax.bar(tags, ppls, color=colours, edgecolor="white", width=0.6)

        for bar, val in zip(bars, ppls):
            label = f"{val:.2f}" if val < 1000 else f"{val:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                label,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_ylabel("Perplexity ↓", fontsize=13)
        ax.set_title(
            "Perplexity (WikiText-2) — Quantisation Comparison",
            fontsize=15, fontweight="bold",
        )
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = self.plots_dir / "perplexity_comparison.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")

        return out
