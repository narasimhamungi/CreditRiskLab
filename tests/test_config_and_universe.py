from datetime import date

import pytest
import yaml

from creditrisklab import config as cfgmod
from creditrisklab.universe import Issuer, UnresolvedUniverseError, cohort_summary, load_issuers, validate_universe


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_all_shipped_configs_load():
    assert cfgmod.load_universe()["defaults"]
    assert cfgmod.load_model_config()["features"]
    assert cfgmod.load_recovery_config()["seniority"]


def test_grade_bounds_must_increase(tmp_path, model_cfg):
    bad = dict(model_cfg)
    bad["grades"] = {"scale": [{"grade": "1", "pd_upper": 0.05}, {"grade": "2", "pd_upper": 0.01}, {"grade": "3", "pd_upper": 1.0}]}
    with pytest.raises(ValueError, match="increasing"):
        cfgmod.load_model_config.__wrapped__(_write(tmp_path, "m.yaml", bad))


def test_final_grade_must_cover_all_pds(tmp_path, model_cfg):
    bad = dict(model_cfg)
    bad["grades"] = {"scale": [{"grade": "1", "pd_upper": 0.01}, {"grade": "2", "pd_upper": 0.5}]}
    with pytest.raises(ValueError, match="1.0"):
        cfgmod.load_model_config.__wrapped__(_write(tmp_path, "m.yaml", bad))


def test_recovery_input_without_citation_tag_is_rejected(tmp_path):
    payload = {"seniority": {"senior_secured": {"recovery_mean": 0.6, "status": "ASSUMED", "citation": "none"}}}
    with pytest.raises(ValueError, match="SOURCED"):
        cfgmod.load_recovery_config.__wrapped__(_write(tmp_path, "r.yaml", payload))


def test_recovery_out_of_range_is_rejected(tmp_path):
    payload = {"seniority": {"senior_secured": {"recovery_mean": 1.4, "basis": "ultimate", "status": "SOURCED", "citation": "x"}}}
    with pytest.raises(ValueError, match="out-of-range"):
        cfgmod.load_recovery_config.__wrapped__(_write(tmp_path, "r.yaml", payload))


def test_shipped_universe_has_ten_defaults_with_complete_exposure_mixes():
    """Structural invariant. Whether events are verified is run state, owned by
    `cli verify-defaults` and the gate test below — not something a test should pin."""
    issuers = load_issuers()
    assert cohort_summary(issuers)["defaults"] == 10
    for issuer in (i for i in issuers if i.is_default):
        assert abs(sum(issuer.exposure.values()) - 1.0) < 1e-9, issuer.ticker


def test_unverified_default_event_blocks_training():
    issuer = Issuer("X", "X", "s", cik="0000000001", default_date=date(2020, 1, 1), exposure={"senior_secured": 1.0})
    with pytest.raises(UnresolvedUniverseError, match="unverified"):
        issuer.validate(require_verified=True)
    issuer.validate(require_verified=False)


def test_exposure_mix_must_sum_to_one():
    issuer = Issuer("X", "X", "s", cik="0000000001", default_date=date(2020, 1, 1), verified=True, exposure={"senior_secured": 0.7})
    with pytest.raises(UnresolvedUniverseError, match="sum"):
        issuer.validate()


def test_missing_cik_blocks_training():
    with pytest.raises(UnresolvedUniverseError, match="CIK"):
        Issuer("X", "X", "s").validate()


def test_recovery_class_must_declare_its_basis(tmp_path):
    payload = {"seniority": {"senior_secured": {"recovery_mean": 0.5, "status": "SOURCED", "citation": "x"}}}
    with pytest.raises(ValueError, match="basis"):
        cfgmod.load_recovery_config.__wrapped__(_write(tmp_path, "r.yaml", payload))
