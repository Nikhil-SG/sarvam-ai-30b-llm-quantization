"""
Tests for MxMoE Module 5: Hub Publication
"""
import json
from pathlib import Path

from src.core.paths import get_mxmoe_module_paths


def _results_dir(config) -> Path:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    return Path(get_mxmoe_module_paths(base_dir, 5)["results_dir"])

def test_publication_report(config):
    res_dir = _results_dir(config)
    fp = res_dir / "publication_report.json"
    
    if getattr(config.deployment.huggingface, "push_to_hub", False):
        assert fp.exists(), "Publication report missing when push_to_hub is True"
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("status") == "success", "Publication did not report success"
    else:
        # If explicitly skipped, it should not fail the test suite, just warn or pass
        pass