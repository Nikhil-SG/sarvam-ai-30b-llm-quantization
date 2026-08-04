import pytest
from src.core.config import load_config

@pytest.fixture
def config():
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'research.yaml')
    return load_config(cfg_path)
