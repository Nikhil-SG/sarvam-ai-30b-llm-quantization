from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


class Module3ReportBuilder:
    """Build publication-friendly summaries for Module 3 outputs."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        cache_validation: Dict[str, Any],
        mse_data: Optional[Dict[str, Any]],
        outlier_runs: Dict[str, Dict[str, Any]],
        module_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = {
            "overview": self._build_overview(cache_validation, mse_data, outlier_runs, module_summary),
            "mse": self._build_mse_summary(mse_data),
            "outliers": self._build_outlier_summary(outlier_runs),
        }
        report["key_findings"] = self._build_findings(report)

        json_path = self.results_dir / "module_3_report.json"
        md_path = self.results_dir / "module_3_report.md"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(self._render_markdown(report))

        return {
            "report": report,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        }

    def _build_overview(
        self,
        cache_validation: Dict[str, Any],
        mse_data: Optional[Dict[str, Any]],
        outlier_runs: Dict[str, Dict[str, Any]],
        module_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_count = 0
        if outlier_runs:
            first = next(iter(outlier_runs.values()))
            prompt_count = int(first.get("num_prompts", 0))

        return {
            "status": module_summary.get("status", "UNKNOWN"),
            "cache_tags": cache_validation.get("tags", []),
            "layers_per_cache_tag": cache_validation.get("layer_counts", {}),
            "mse_quantizers": [] if not mse_data else mse_data.get("quantizer_tags", []),
            "outlier_tags": list(outlier_runs.keys()),
            "prompt_count": prompt_count,
        }

    def _build_mse_summary(self, mse_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not mse_data:
            return {"ranking": [], "per_quantizer": {}}

        per_quantizer: Dict[str, Any] = {}
        for tag, raw in mse_data.get("raw_data", {}).items():
            if not raw:
                continue
            items = list(raw.items())
            values = [float(value) for _, value in items]
            worst_layer, worst_value = max(items, key=lambda kv: kv[1])
            best_layer, best_value = min(items, key=lambda kv: kv[1])
            per_quantizer[tag] = {
                "mean_mse": float(mean(values)),
                "min_mse": float(best_value),
                "max_mse": float(worst_value),
                "best_layer": best_layer,
                "worst_layer": worst_layer,
                "num_points": len(values),
            }

        ranking = sorted(
            (
                {"tag": tag, **stats}
                for tag, stats in per_quantizer.items()
            ),
            key=lambda item: item["mean_mse"],
        )
        return {"ranking": ranking, "per_quantizer": per_quantizer}

    def _build_outlier_summary(self, outlier_runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        per_tag: Dict[str, Any] = {}
        for tag, result in outlier_runs.items():
            stats = result.get("statistics", {})
            if not stats:
                continue
            items = list(stats.items())
            mean_pct = float(mean(item[1]["outlier_pct"] for item in items))
            mean_absmax = float(mean(item[1]["abs_max"] for item in items))
            highest_outlier_layer, highest_outlier_stats = max(
                items,
                key=lambda item: item[1]["outlier_pct"],
            )
            highest_abs_layer, highest_abs_stats = max(
                items,
                key=lambda item: item[1]["abs_max"],
            )
            per_tag[tag] = {
                "mean_outlier_pct": mean_pct,
                "mean_abs_max": mean_absmax,
                "highest_outlier_layer": highest_outlier_layer,
                "highest_outlier_pct": float(highest_outlier_stats["outlier_pct"]),
                "highest_absmax_layer": highest_abs_layer,
                "highest_absmax": float(highest_abs_stats["abs_max"]),
                "num_layers": len(items),
                "num_prompts": int(result.get("num_prompts", 0)),
            }

        ranking = sorted(
            (
                {"tag": tag, **stats}
                for tag, stats in per_tag.items()
            ),
            key=lambda item: item["mean_outlier_pct"],
            reverse=True,
        )
        return {"ranking": ranking, "per_tag": per_tag}

    def _build_findings(self, report: Dict[str, Any]) -> List[str]:
        findings: List[str] = []

        mse_ranking = report["mse"].get("ranking", [])
        if mse_ranking:
            best = mse_ranking[0]
            worst = mse_ranking[-1]
            findings.append(
                f"Lowest average weight error is {best['tag'].upper()} ({best['mean_mse']:.3e}), while the highest is {worst['tag'].upper()} ({worst['mean_mse']:.3e})."
            )
            findings.append(
                f"The largest single MSE hotspot appears in {worst['tag'].upper()} at {worst['worst_layer']} ({worst['max_mse']:.3e})."
            )

        outlier_ranking = report["outliers"].get("ranking", [])
        if outlier_ranking:
            highest = outlier_ranking[0]
            findings.append(
                f"Highest average activation outlier rate is {highest['tag'].upper()} at {highest['mean_outlier_pct']:.4f}% across {highest['num_prompts']} prompts."
            )
            bf16 = report["outliers"]["per_tag"].get("bf16")
            if bf16:
                findings.append(
                    f"BF16 baseline shows its strongest activation hotspot at {bf16['highest_outlier_layer']} ({bf16['highest_outlier_pct']:.4f}% outliers), which is the reference point for quantized comparisons."
                )

        if not findings:
            findings.append("Module 3 completed without enough data to derive ranked findings.")

        return findings

    def _render_markdown(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        overview = report["overview"]
        lines.append("# Module 3 Research Summary")
        lines.append("")
        lines.append(f"Status: {overview['status']}")
        lines.append(f"Cache tags: {', '.join(overview['cache_tags']) if overview['cache_tags'] else 'none'}")
        lines.append(f"MSE quantizers: {', '.join(overview['mse_quantizers']) if overview['mse_quantizers'] else 'none'}")
        lines.append(f"Outlier tags: {', '.join(overview['outlier_tags']) if overview['outlier_tags'] else 'none'}")
        lines.append(f"Prompt count: {overview['prompt_count']}")
        lines.append("")
        lines.append("## Key Findings")
        lines.append("")
        for finding in report["key_findings"]:
            lines.append(f"- {finding}")

        mse_ranking = report["mse"].get("ranking", [])
        if mse_ranking:
            lines.append("")
            lines.append("## MSE Ranking")
            lines.append("")
            lines.append("| Quantizer | Mean MSE | Worst Layer | Worst MSE |")
            lines.append("| --- | ---: | --- | ---: |")
            for row in mse_ranking:
                lines.append(
                    f"| {row['tag'].upper()} | {row['mean_mse']:.3e} | {row['worst_layer']} | {row['max_mse']:.3e} |"
                )

        outlier_ranking = report["outliers"].get("ranking", [])
        if outlier_ranking:
            lines.append("")
            lines.append("## Outlier Ranking")
            lines.append("")
            lines.append("| Tag | Mean Outlier % | Highest-Outlier Layer | Highest-Outlier % |")
            lines.append("| --- | ---: | --- | ---: |")
            for row in outlier_ranking:
                lines.append(
                    f"| {row['tag'].upper()} | {row['mean_outlier_pct']:.4f} | {row['highest_outlier_layer']} | {row['highest_outlier_pct']:.4f} |"
                )

        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append("This report combines weight-space distortion and activation-space instability so the quantization tradeoff can be read as both an engineering and model-behavior story.")
        lines.append("")
        return "\n".join(lines)