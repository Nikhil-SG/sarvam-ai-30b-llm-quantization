#!/usr/bin/env python3
"""
Diagnostic: Verify weight cache integrity across quantisation formats.

1. Loads cached .npz weight files for bf16, fp8, int8, nf4, gptq
2. Compares values between formats to detect near-identical caches
3. Reports per-format statistics (range, std, MSE vs BF16)
4. Identifies which formats need re-extraction

Usage:
    python check_weight_caches.py
    python check_weight_caches.py --purge   # also delete bad caches
"""

import argparse
import numpy as np
import shutil
from pathlib import Path

SHARED_DIR = Path("./research/outputs/shared_weights")
TAGS = ["bf16", "fp8", "int8", "nf4", "gptq"]

# Pick representative layers spanning early, middle, late
CHECK_LAYERS = [
    "model_layers_0_self_attn_q_proj",
    "model_layers_0_mlp_down_proj",
    "model_layers_40_self_attn_q_proj",
    "model_layers_79_mlp_down_proj",
]


def pearson_r(a, b):
    """Pearson correlation coefficient."""
    am, bm = a.mean(), b.mean()
    ad, bd = a - am, b - bm
    denom = np.sqrt((ad ** 2).sum() * (bd ** 2).sum())
    if denom == 0:
        return 1.0
    return float((ad * bd).sum() / denom)


def suspicious_thresholds(tag: str):
    """Per-format near-identity thresholds for BF16 comparison."""
    # LLM.int8() is intentionally near-lossless; use stricter thresholds
    # to avoid false positives in the standalone checker.
    if tag == "int8":
        return 0.999999, 0.001
    return 0.9999, 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge", action="store_true",
                        help="Delete caches deemed near-identical to BF16")
    args = parser.parse_args()

    print("=" * 72)
    print("  Weight Cache Integrity Checker")
    print("=" * 72)

    if not SHARED_DIR.exists():
        print(f"\n  ERROR: {SHARED_DIR} not found.")
        return

    # Discover which tags have caches
    available = {}
    for tag in TAGS:
        tag_dir = SHARED_DIR / tag
        if tag_dir.is_dir():
            npz_count = len(list(tag_dir.glob("*.npz")))
            available[tag] = npz_count
            print(f"  {tag:6s}: {npz_count} cached layers")
        else:
            print(f"  {tag:6s}: NOT FOUND")

    if "bf16" not in available:
        print("\n  ERROR: No BF16 baseline cache. Cannot validate.")
        return

    quant_tags = [t for t in available if t != "bf16"]
    if not quant_tags:
        print("\n  No quantised caches to check.")
        return

    # Load and compare for representative layers
    suspect_tags = set()

    for layer_name in CHECK_LAYERS:
        bf16_path = SHARED_DIR / "bf16" / f"{layer_name}.npz"
        if not bf16_path.exists():
            continue

        bf16_data = np.load(bf16_path)
        bf16_vals = bf16_data["values"].astype(np.float64)

        print(f"\n{'─' * 72}")
        print(f"  Layer: {layer_name}")
        print(f"  BF16: min={bf16_vals.min():.6f}  max={bf16_vals.max():.6f}  "
              f"std={bf16_vals.std():.6f}  samples={len(bf16_vals)}")

        for tag in quant_tags:
            tag_path = SHARED_DIR / tag / f"{layer_name}.npz"
            if not tag_path.exists():
                print(f"  {tag:6s}: MISSING")
                continue

            tag_data = np.load(tag_path)
            tag_vals = tag_data["values"].astype(np.float64)

            if len(tag_vals) != len(bf16_vals):
                print(f"  {tag:6s}: SIZE MISMATCH ({len(tag_vals)} vs {len(bf16_vals)})")
                continue

            # Statistics
            diff = bf16_vals - tag_vals
            mse = float((diff ** 2).mean())
            max_diff = float(np.abs(diff).max())
            r = pearson_r(bf16_vals, tag_vals)

            # Range of the quantised weights
            t_min, t_max = float(tag_vals.min()), float(tag_vals.max())
            t_std = float(tag_vals.std())

            # Verdict
            r_thresh, d_thresh = suspicious_thresholds(tag)
            if r > r_thresh and max_diff < d_thresh:
                verdict = "⚠ SUSPECT (near-identical to BF16!)"
                suspect_tags.add(tag)
            elif tag == "int8" and r > 0.9999 and max_diff < 0.05:
                verdict = "? POSSIBLY OK (INT8 can be near-lossless)"
            elif r > 0.999:
                verdict = "? POSSIBLY OK (high correlation)"
            else:
                verdict = "✓ OK (distinctly different from BF16)"

            print(f"  {tag:6s}: min={t_min:.6f}  max={t_max:.6f}  std={t_std:.6f}")
            print(f"          MSE={mse:.8f}  max_diff={max_diff:.6f}  "
                  f"Pearson_r={r:.10f}")
            print(f"          → {verdict}")

    # Cross-format comparison (FP8 vs INT8 vs NF4)
    print(f"\n{'═' * 72}")
    print("  Cross-Format Comparison (should differ for different quant methods)")
    print(f"{'═' * 72}")

    pairs = [("fp8", "int8"), ("fp8", "nf4"), ("int8", "nf4")]
    for tag_a, tag_b in pairs:
        if tag_a not in available or tag_b not in available:
            continue
        for layer_name in CHECK_LAYERS[:1]:  # just first layer
            pa = SHARED_DIR / tag_a / f"{layer_name}.npz"
            pb = SHARED_DIR / tag_b / f"{layer_name}.npz"
            if not (pa.exists() and pb.exists()):
                continue
            va = np.load(pa)["values"].astype(np.float64)
            vb = np.load(pb)["values"].astype(np.float64)
            if len(va) != len(vb):
                continue
            r = pearson_r(va, vb)
            mse = float(((va - vb) ** 2).mean())
            md = float(np.abs(va - vb).max())
            verdict = ("⚠ IDENTICAL" if r > 0.9999 and md < 0.01
                       else "✓ DIFFERENT" if r < 0.99
                       else "? SUSPICIOUS")
            print(f"  {tag_a} vs {tag_b}: r={r:.10f}  MSE={mse:.8f}  "
                  f"max_diff={md:.6f}  → {verdict}")

    # Summary
    print(f"\n{'═' * 72}")
    if suspect_tags:
        print(f"  PROBLEM: These caches are near-identical to BF16 under")
        print(f"  the configured thresholds: {', '.join(sorted(suspect_tags))}")
        print()
        if suspect_tags == {"int8"}:
            print("  Note: INT8 (LLM.int8) can be near-lossless by design.")
            print("  Confirm cached checkpoint reload is truly quantized")
            print("  (Linear8bitLt/int8 weights) before purging.")
        else:
            print("  This can indicate dequantization fallback issues.")
            print("  Confirm checkpoint reload path and model module types.")
        print()
        if args.purge:
            for tag in sorted(suspect_tags):
                cache_dir = SHARED_DIR / tag
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    print(f"  PURGED: {cache_dir}")
            print()
            print("  Re-run Module 2 to regenerate with fixed extraction:")
            print("    python main.py --module 2")
        else:
            print("  To purge and regenerate: python check_weight_caches.py --purge")
            print("  Then re-run: python main.py --module 2")
    else:
        print("  All caches look correctly dequantised. ✓")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
