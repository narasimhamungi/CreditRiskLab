import shutil
from datetime import date

import pandas as pd
import yaml

from creditrisklab.config import config_dir, load_universe
from creditrisklab.universe import default_event_confirmed, update_universe_file


def _copy(tmp_path):
    target = tmp_path / "universe.yaml"
    shutil.copy(config_dir() / "universe.yaml", target)
    return target


def test_unquoted_zero_padded_cik_is_an_octal_trap():
    """Why every CIK is written quoted: YAML 1.1 reads Frontier's CIK as octal."""
    assert yaml.safe_load("cik: 0000020520")["cik"] != "0000020520"
    assert yaml.safe_load('cik: "0000020520"')["cik"] == "0000020520"


def test_block_entry_fields_are_updated_and_comments_survive(tmp_path):
    path = _copy(tmp_path)
    update_universe_file(path, {"RAD": {"cik": "0000084129", "verified": True}})
    text = path.read_text(encoding="utf-8")
    assert "# FACT DISCIPLINE" in text and "# used for EAD/LGD segmentation" in text
    row = next(r for r in yaml.safe_load(text)["defaults"] if r["ticker"] == "RAD")
    assert row["cik"] == "0000084129" and row["verified"] is True
    load_universe.cache_clear()


def test_update_touches_only_the_named_ticker(tmp_path):
    path = _copy(tmp_path)
    before = {r["ticker"]: r["verified"] for r in yaml.safe_load(path.read_text(encoding="utf-8"))["defaults"]}
    update_universe_file(path, {"JCP": {"verified": True}})
    after = {r["ticker"]: r["verified"] for r in yaml.safe_load(path.read_text(encoding="utf-8"))["defaults"]}
    assert after["JCP"] is True
    assert {k: v for k, v in after.items() if k != "JCP"} == {k: v for k, v in before.items() if k != "JCP"}
    load_universe.cache_clear()


def test_flow_style_entry_is_updated(tmp_path):
    path = _copy(tmp_path)
    update_universe_file(path, {"NKE": {"cik": "0000320187"}})
    row = next(r for r in yaml.safe_load(path.read_text(encoding="utf-8"))["non_defaults"] if r["ticker"] == "NKE")
    assert row["cik"] == "0000320187"
    load_universe.cache_clear()


def _filings(rows):
    return pd.DataFrame(rows, columns=["form", "filingDate", "items"])


def test_item_1_03_inside_window_confirms_the_event():
    f = _filings([("8-K", "2020-05-18", "1.03,7.01,9.01"), ("8-K", "2020-03-01", "2.02")])
    ok, dates = default_event_confirmed(f, date(2020, 5, 15), 30)
    assert ok and dates == ["2020-05-18"]


def test_item_1_03_outside_window_does_not_confirm():
    f = _filings([("8-K", "2021-01-30", "1.03,2.01")])
    assert default_event_confirmed(f, date(2020, 5, 15), 30)[0] is False


def test_other_items_do_not_confirm():
    f = _filings([("8-K", "2020-05-16", "1.01,2.03,9.01"), ("8-K", "2020-05-17", "10.3")])
    assert default_event_confirmed(f, date(2020, 5, 15), 30)[0] is False


def test_shipped_ciks_are_strings_of_ten_digits():
    for section in ("defaults", "non_defaults"):
        for row in load_universe()[section]:
            assert isinstance(row["cik"], str) and len(row["cik"]) == 10 and row["cik"].isdigit()
