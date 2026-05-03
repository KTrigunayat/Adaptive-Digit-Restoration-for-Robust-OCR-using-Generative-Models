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
        "distortion": "gaussian_noise",  # 'gaussian_noise' | 'motion_blur' | 'spatial_masking'
        "seed": 42,
    },
    "preprocessing": {
        "filter": "median",
        "kernel_size": 3,
    },
    "vae": {
        "latent_dim": 64,
        "beta": 1.0,
        "epochs": 20,
        "lr": 1e-4,
        "checkpoint": "checkpoints/vae.pth",
    },
    "corruption_classifier": {
        "epochs": 15,
        "lr": 1e-4,
        "checkpoint": "checkpoints/corruption_classifier.pth",
    },
    "diffusion": {
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "epochs": 30,
        "lr": 1e-4,
        "checkpoint": "checkpoints/diffusion.pth",
    },
    "ocr": {
        "epochs": 10,
        "lr": 1e-3,
        "checkpoint": "checkpoints/ocr.pth",
    },
    "evaluation": {
        "output_path": "experiments/results/eval_report.json",
    },
    "device": "cuda",
    "seed": 42,
}

# Required keys that must be present in the final config (dot-notation).
_REQUIRED_KEYS = [
    "data.raw_dir",
    "vae.latent_dim",
    "diffusion.timesteps",
    "diffusion.beta_start",
    "diffusion.beta_end",
]


def _get_nested(cfg: dict, dotted_key: str):
    """Return the value at a dot-separated key path, or raise KeyError."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML, falling back to defaults for missing keys.

    Raises:
        KeyError: If a required configuration key is absent from the final config.
    """
    # Deep-copy defaults so mutations don't bleed between calls.
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    cfg_path = Path(path)
    if cfg_path.exists():
        with open(cfg_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values

    # Validate required keys.
    for key in _REQUIRED_KEYS:
        _get_nested(config, key)  # raises KeyError(key) if missing

    return config
