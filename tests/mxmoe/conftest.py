import pytest
from src.core.config import load_config
from src.core.paths import scope_config_to_mxmoe_module

@pytest.fixture
def config():
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'mxmoe.yaml')
    return load_config(cfg_path)
