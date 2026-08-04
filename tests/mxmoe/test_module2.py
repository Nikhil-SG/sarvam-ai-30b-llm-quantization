"""
Tests for MxMoE Module 2: Mixed-Precision Synthesis
"""
import json
from pathlib import Path

from src.core.paths import get_mxmoe_module_paths
from tests.test_runner import SkipTest


def _results_dir(config) -> Path:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    return Path(get_mxmoe_module_paths(base_dir, 2)["results_dir"])


def _quantized_models_dir(config) -> Path:
    out_cfg = getattr(config, "output", None)
    return Path(getattr(out_cfg, "quantized_models_dir", "mxmoe/quantized_models"))


def _require_path(path: Path, module_label: str) -> None:
    if not path.exists():
        raise SkipTest(f"{module_label} artifacts missing: run the module first ({path})")

def test_precision_recipe(config):
    res_dir = _results_dir(config)
    fp = res_dir / "precision_recipe.json"
    _require_path(fp, "Module 2")
    assert fp.exists(), f"Precision recipe missing at: {fp}"
    
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "fp8_targets" in data, "fp8_targets array missing"
    assert "gptq_targets" in data, "gptq_targets array missing"
    assert len(data["fp8_targets"]) > 0, "fp8_targets should not be empty"

    # int8_targets should be populated with same modules as fp8_targets
    assert "int8_targets" in data, "int8_targets array missing"
    assert len(data["int8_targets"]) > 0, "int8_targets should not be empty"
    assert len(data["int8_targets"]) == len(data["fp8_targets"]), (
        f"int8_targets ({len(data['int8_targets'])}) should match "
        f"fp8_targets ({len(data['fp8_targets'])})"
    )

    # gptq_low_targets is legacy — should be empty
    assert len(data.get("gptq_low_targets", [])) == 0, "gptq_low_targets should be empty (legacy)"

    # Verify metadata has strategies
    metadata = data.get("metadata", {})
    strategies = metadata.get("strategies", [])
    assert len(strategies) > 0, "metadata.strategies should list active strategies"


def test_compressed_model_exists(config):
    """Check that at least one strategy-specific quantized model directory exists."""
    save_dir = _quantized_models_dir(config)
    strategies = ["fp8_gptq", "int8_gptq"]

    found_any = False
    for strategy in strategies:
        strategy_dir = save_dir.parent / f"{save_dir.name}_{strategy}"
        if strategy_dir.exists():
            safetensors = list(strategy_dir.glob("*.safetensors"))
            if len(safetensors) > 0:
                assert (strategy_dir / "config.json").exists(), \
                    f"Model config.json missing in {strategy_dir}"
                found_any = True

    # Also check the base dir for backward compatibility
    if save_dir.exists():
        safetensors = list(save_dir.glob("*.safetensors"))
        if len(safetensors) > 0:
            found_any = True

    if not found_any:
        raise SkipTest(
            f"Module 2 artifacts missing: no quantized models found in "
            f"{save_dir} or strategy-specific dirs; run module 2 first"
        )


def test_compressor_uses_config_calibration_dataset(config):
    from src.mxmoe.recipe.compressor import ModelCompressor

    compressor = ModelCompressor(config)
    cal_cfg = config.recipe.calibration

    assert compressor.calib_dataset == getattr(cal_cfg, "dataset", None)
    assert compressor.calib_dataset_config == getattr(cal_cfg, "dataset_config", None)
    assert compressor.calib_split == getattr(cal_cfg, "split", "train")


def test_recipe_builder_strategies(config):
    """Verify RecipeBuilder picks up strategies from config."""
    from src.mxmoe.recipe.recipe_builder import RecipeBuilder

    builder = RecipeBuilder(config)
    assert hasattr(builder, "strategies"), "RecipeBuilder should have strategies attribute"
    assert "fp8_gptq" in builder.strategies, "fp8_gptq should be in strategies"
    assert "int8_gptq" in builder.strategies, "int8_gptq should be in strategies"