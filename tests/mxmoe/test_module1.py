"""
Tests for MxMoE Module 1: Sensitivity-Aware Profiling
Ensures Fisher Scores, Routing Stats, and Importance Map are generated correctly.
"""
import json
from pathlib import Path

from src.core.paths import get_mxmoe_module_paths
from tests.test_runner import SkipTest


def _results_dir(config) -> Path:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    return Path(get_mxmoe_module_paths(base_dir, 1)["results_dir"])


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SkipTest(f"Module 1 artifacts missing: run MxMoE module 1 first ({path})")

def test_fisher_scores_exist_and_valid(config):
    res_dir = _results_dir(config)
    fp = res_dir / "fisher_scores.json"
    _require_file(fp)
    assert fp.exists(), f"Fisher scores missing at: {fp}"
    
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "fisher_scores" in data, "fisher_scores key missing in json"
    assert len(data["fisher_scores"]) > 0, "No layers logged in fisher scores"

def test_routing_stats_exist(config):
    res_dir = _results_dir(config)
    fp = res_dir / "routing_stats.json"
    _require_file(fp)
    assert fp.exists(), f"Routing stats missing at: {fp}"
    
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "routing_frequency" in data, "routing_frequency key missing"

def test_importance_map_format(config):
    res_dir = _results_dir(config)
    fp = res_dir / "expert_importance_map.json"
    _require_file(fp)
    assert fp.exists(), f"Importance map missing at: {fp}"
    
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    core = data.get("importance_map", data)
    assert len(core) >= 1, "Importance map is empty"
    
    # Check a sampled layer
    sample_layer = list(core.keys())[0]
    sample_expert = list(core[sample_layer].keys())[0]
    
    expected_keys = ["importance", "recommended_precision"]
    for k in expected_keys:
        assert k in core[sample_layer][sample_expert], f"Missing '{k}' in importance map expert data"


def test_fisher_analyzer_uses_config_calibration_dataset(config):
    from src.mxmoe.sensitivity.fisher_info import FisherInformationAnalyzer

    analyzer = FisherInformationAnalyzer(config)
    cal_cfg = config.sensitivity.calibration

    assert analyzer.calib_dataset == getattr(cal_cfg, "dataset", None)
    assert analyzer.calib_dataset_config == getattr(cal_cfg, "dataset_config", None)
    assert analyzer.calib_split == getattr(cal_cfg, "split", "train")