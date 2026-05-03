"""
config.py
Loads and validates pipeline configuration from a YAML file.
"""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "distortion": "noise",          # 'noise' | 'blur' | 'mask'
        "distortion_var": 0.05,
    },
    "preprocessing": {
        "filter": "gaussian",           # 'gaussian' | 'median'
        "kernel_size": 5,
    },
    "vae": {
        "latent_dim": 64,
        "epochs": 20,
        "lr": 1e-3,
    },
    "diffusion": {
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "epochs": 30,
        "lr": 2e-4,
    },
    "ocr": {
        "epochs": 10,
        "lr": 1e-3,
    },
    "device": "cuda",
}


def load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML, falling back to defaults for missing keys."""
    config = DEFAULT_CONFIG.copy()
    cfg_path = Path(path)
    if cfg_path.exists():
        with open(cfg_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values
    return config
