"""
Artifact contract for cross-pipeline data sharing.

Defines where Research pipeline outputs live and how MxMoE can locate
and validate them.  No fancy serialization — just structured path lookups
with existence checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class ResearchArtifacts:
    """Locate and validate artifacts produced by the Research pipeline."""

    def __init__(self, research_output_dir: str = "research/outputs"):
        self.base = Path(research_output_dir)

    # ── Path helpers ─────────────────────────────────────────────────────

    @property
    def bf16_results(self) -> Path:
        return self.base / "module_1_baseline" / "results" / "bf16_results.json"

    @property
    def weight_cache_dir(self) -> Path:
        return self.base / "shared_weights"

    @property
    def quantization_results_dir(self) -> Path:
        return self.base / "module_2_quantization" / "results"

    @property
    def perplexity_results(self) -> Path:
        return self.base / "module_5_evaluation" / "results" / "perplexity_results.json"

    @property
    def benchmark_results(self) -> Path:
        return self.base / "module_5_evaluation" / "results" / "benchmark_results.json"

    @property
    def pipeline_summary(self) -> Path:
        return self.base / "pipeline_summary.json"

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self, require_all: bool = False) -> List[str]:
        """
        Return a list of issues (empty = valid).

        Args:
            require_all: If True, checks all artifacts (strict).
                         If False, only checks the minimum needed (BF16 baseline).
        """
        issues: List[str] = []

        if not self.base.exists():
            issues.append(f"Research output directory not found: {self.base}")
            return issues

        if not self.bf16_results.exists():
            issues.append(f"Missing BF16 baseline: {self.bf16_results}")

        if not self.weight_cache_dir.exists():
            issues.append(f"Missing weight cache: {self.weight_cache_dir}")

        if require_all:
            if not self.perplexity_results.exists():
                issues.append(f"Missing perplexity results: {self.perplexity_results}")
            if not self.benchmark_results.exists():
                issues.append(f"Missing benchmark results: {self.benchmark_results}")

        return issues

    # ── Data loaders ─────────────────────────────────────────────────────

    def load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a JSON artifact, returning None if missing."""
        if not path.exists():
            logger.warning(f"Artifact not found: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to load artifact {path}: {exc}")
            return None

    def load_bf16_perplexity(self) -> Optional[float]:
        """Load BF16 perplexity as the quality ceiling for MxMoE."""
        data = self.load_json(self.perplexity_results)
        if data is None:
            return None
        return data.get("bf16", {}).get("perplexity")

    def load_quantizer_results(self) -> Dict[str, Any]:
        """Load all quantizer result JSONs from Module 2."""
        results = {}
        qdir = self.quantization_results_dir
        if not qdir.exists():
            return results
        for fp in qdir.glob("*_results.json"):
            method = fp.stem.replace("_results", "")
            data = self.load_json(fp)
            if data is not None:
                results[method] = data
        return results
