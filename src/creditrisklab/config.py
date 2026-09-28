"""Configuration loading.

Every numeric assumption in this project lives in `config/*.yaml`, not in code. The
loaders below validate structure on read so a malformed config fails at startup rather
than silently producing a plausible-looking wrong number three stages later.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Where data/ (raw EDGAR cache, interim, outputs) lives.

    CREDITRISKLAB_ROOT overrides it (tests use this). In a source checkout it is the repo
    root; in a plain pip install -- where two folders above this file is the virtualenv's
    Lib/ directory, not a project -- it is ~/.creditrisklab. The old unconditional
    parents[2] rule only ever worked from a checkout: a pinned install from a git tag
    (found by Keystone's clean install) looked for config under .venv/Lib/config."""
    env = os.environ.get("CREDITRISKLAB_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").exists():
        return repo
    return Path.home() / ".creditrisklab"


def config_dir() -> Path:
    """The model configuration (training universe, features, recovery rates) is part of
    the model, so it ships INSIDE the package and is pinned with it. CREDITRISKLAB_ROOT, when
    set, still points at <root>/config (tests copy the config there and edit it)."""
    env = os.environ.get("CREDITRISKLAB_ROOT")
    if env:
        return Path(env) / "config"
    return Path(__file__).resolve().parent / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} did not parse to a mapping")
    return data


@lru_cache(maxsize=None)
def load_universe(path: str | None = None) -> dict[str, Any]:
    data = _read_yaml(Path(path) if path else config_dir() / "universe.yaml")
    for key in ("meta", "defaults", "non_defaults"):
        if key not in data:
            raise ValueError(f"universe.yaml missing required key: {key}")
    if not data["defaults"]:
        raise ValueError("universe.yaml declares no default events; a PD model needs ground truth")
    return data


@lru_cache(maxsize=None)
def load_model_config(path: str | None = None) -> dict[str, Any]:
    data = _read_yaml(Path(path) if path else config_dir() / "model.yaml")
    for key in ("features", "logistic", "calibration", "validation", "grades"):
        if key not in data:
            raise ValueError(f"model.yaml missing required key: {key}")
    _validate_grades(data["grades"])
    return data


@lru_cache(maxsize=None)
def load_recovery_config(path: str | None = None) -> dict[str, Any]:
    data = _read_yaml(Path(path) if path else config_dir() / "recovery_rates.yaml")
    if "seniority" not in data:
        raise ValueError("recovery_rates.yaml missing 'seniority'")
    for name, block in data["seniority"].items():
        if block.get("status") != "SOURCED":
            raise ValueError(
                f"recovery class '{name}' is not tagged SOURCED. Recovery inputs must carry "
                "a citation; untagged figures are not permitted."
            )
        if "citation" not in block:
            raise ValueError(f"recovery class '{name}' has no citation")
        if block.get("basis") not in {"trading_price", "ultimate"}:
            raise ValueError(f"recovery class '{name}' must declare basis: trading_price | ultimate")
        r = block.get("recovery_mean")
        if r is None or not 0.0 <= float(r) <= 1.0:
            raise ValueError(f"recovery class '{name}' has an out-of-range recovery_mean")
    return data


def _validate_grades(grades: dict[str, Any]) -> None:
    scale = grades.get("scale")
    if not scale:
        raise ValueError("model.yaml grades.scale is empty")
    uppers = [float(g["pd_upper"]) for g in scale]
    if uppers != sorted(uppers):
        raise ValueError("grade PD upper bounds must be strictly increasing")
    if len(set(uppers)) != len(uppers):
        raise ValueError("grade PD upper bounds must be unique")
    if abs(uppers[-1] - 1.0) > 1e-9:
        raise ValueError("the final grade must have pd_upper = 1.0 so every PD maps to a grade")


def feature_names(model_cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = model_cfg or load_model_config()
    return [f["name"] for f in cfg["features"]]
