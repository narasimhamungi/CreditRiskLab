"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import pytest

from creditrisklab.config import load_model_config, load_recovery_config
from creditrisklab.features.panel import build_panel
from creditrisklab.synthetic import build_fundamentals, stamped_issuers
from creditrisklab.universe import Issuer, load_issuers


@pytest.fixture(scope="session")
def model_cfg() -> dict:
    return load_model_config()


@pytest.fixture(scope="session")
def fast_cfg() -> dict:
    """Same model, fewer resamples — keeps the suite fast without changing any logic."""
    cfg = copy.deepcopy(load_model_config())
    cfg["validation"]["cv_repeats"] = 3
    cfg["validation"]["bootstrap_iterations"] = 200
    cfg["gbm"]["max_iter"] = 40
    return cfg


@pytest.fixture(scope="session")
def recovery_cfg() -> dict:
    return load_recovery_config()


@pytest.fixture(scope="session")
def issuers() -> list[Issuer]:
    return stamped_issuers(load_issuers())


@pytest.fixture(scope="session")
def fundamentals(issuers) -> dict[str, pd.DataFrame]:
    return build_fundamentals(issuers, seed=0)


@pytest.fixture(scope="session")
def panel(issuers, fundamentals, model_cfg) -> pd.DataFrame:
    return build_panel(issuers, fundamentals, model_cfg=model_cfg)


@pytest.fixture(scope="session")
def run_result(panel, issuers, fast_cfg, recovery_cfg):
    from creditrisklab.pipeline import run

    return run(panel, issuers, model_cfg=fast_cfg, recovery_cfg=recovery_cfg, split_year=2021, portfolio_date=date(2019, 6, 30))


def long_facts(rows: list[tuple]) -> pd.DataFrame:
    """Build a long fundamentals frame from (field, period_end, filed, value[, tag]) tuples."""
    records = []
    for row in rows:
        field, period_end, filed, value = row[:4]
        tag = row[4] if len(row) > 4 else f"T_{field}"
        records.append(
            {
                "cik": "0000000001",
                "ticker": "TST",
                "field": field,
                "period_end": period_end,
                "filed": filed,
                "form": "10-K",
                "fiscal_year": period_end.year,
                "value": value,
                "tag": tag,
                "source": "test",
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def make_facts():
    return long_facts
