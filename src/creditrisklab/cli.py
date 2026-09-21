"""CreditRiskLab command line.

    python -m creditrisklab.cli demo              # offline, synthetic, watermarked
    python -m creditrisklab.cli resolve           # resolve + verify CIKs against EDGAR
    python -m creditrisklab.cli verify-defaults   # check bankruptcy 8-Ks against claimed dates
    python -m creditrisklab.cli ingest            # pull 10-K XBRL facts for the universe
    python -m creditrisklab.cli run               # full pipeline on real data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from creditrisklab.config import load_model_config, load_recovery_config, load_universe, project_root
from creditrisklab.features.panel import build_panel
from creditrisklab.pipeline import run as run_pipeline
from creditrisklab.reporting import charts
from creditrisklab.reporting import model_validation_report as report
from creditrisklab.universe import UnresolvedUniverseError, cohort_summary, load_issuers, validate_universe

INTERIM = "data/interim"
OUTPUTS = "data/outputs"


def _out(path: str) -> Path:
    full = project_root() / path
    full.mkdir(parents=True, exist_ok=True)
    return full


def cmd_demo(args: argparse.Namespace) -> int:
    from creditrisklab.synthetic import SYNTHETIC_BANNER, build_fundamentals, stamped_issuers

    issuers = stamped_issuers(load_issuers())
    fundamentals = build_fundamentals(issuers, seed=args.seed)
    panel = build_panel(issuers, fundamentals)
    result = run_pipeline(panel, issuers, split_year=args.split_year, portfolio_date=_date(args.portfolio_date))

    out_dir = _out(OUTPUTS)
    panel.to_csv(out_dir / "panel_synthetic.csv", index=False)
    content = SYNTHETIC_BANNER + "\n\n" + report.render(result, load_recovery_config())
    (out_dir / "MODEL_VALIDATION_SYNTHETIC.md").write_text(content, encoding="utf-8")
    charts.build_all(result, out_dir / "charts_synthetic")

    print(SYNTHETIC_BANNER.replace("> ", "").replace("**", ""))
    _print_summary(result)
    print(f"\nwrote {out_dir/'MODEL_VALIDATION_SYNTHETIC.md'}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    """Verify every CIK against EDGAR's registrant name (current or former) and write the
    verified ones into config/universe.yaml. A CIK is written only after the name matches."""
    from creditrisklab.config import config_dir
    from creditrisklab.ingest.edgar_client import resolve_cik_by_ticker, verify_cik
    from creditrisklab.universe import update_universe_file

    issuers = load_issuers()
    rows, updates = [], {}
    for issuer in issuers:
        cik, method = issuer.cik, "config"
        if cik is None:
            cik, method = resolve_cik_by_ticker(issuer.ticker), "ticker map"
        ok, name = False, ""
        if cik:
            try:
                ok, name = verify_cik(cik, issuer.edgar_name_query or issuer.name)
            except Exception as exc:  # noqa: BLE001
                name = f"error: {exc}"
        if ok and issuer.cik != cik:
            updates[issuer.ticker] = {"cik": cik}
        rows.append({"ticker": issuer.ticker, "cik": cik, "method": method, "edgar_name": name, "verified": ok})

    frame = pd.DataFrame(rows)
    path = _out(INTERIM) / "cik_resolution.csv"
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    if updates and not args.no_write:
        for change in update_universe_file(config_dir() / "universe.yaml", updates):
            print(f"  wrote {change}")
    failed = frame.loc[~frame["verified"], "ticker"].tolist()
    if failed:
        print(f"\nNOT VERIFIED: {failed}. Find the CIK on EDGAR company search, put it in "
              "config/universe.yaml, and re-run. Unverified CIKs are never used.")
        return 1
    print("\nall CIKs verified")
    return 0


def cmd_verify_defaults(args: argparse.Namespace) -> int:
    """Confirm each default date against an Item 1.03 (bankruptcy) 8-K filed within
    +/- `window` days. Confirmed events are marked verified: true in universe.yaml; anything
    unconfirmed stays false and blocks training until reviewed."""
    from creditrisklab.config import config_dir
    from creditrisklab.ingest.edgar_client import list_filings
    from creditrisklab.universe import default_event_confirmed, update_universe_file

    rows, updates = [], {}
    for issuer in [i for i in load_issuers() if i.is_default]:
        if not issuer.cik:
            rows.append({"ticker": issuer.ticker, "confirmed": False, "item_1_03_8k_dates": "", "note": "CIK unresolved — run resolve first"})
            continue
        try:
            filings = list_filings(issuer.cik, form="8-K", around=issuer.default_date, window_days=args.window + 30)
            ok, dates = default_event_confirmed(filings, issuer.default_date, args.window)
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": issuer.ticker, "confirmed": False, "item_1_03_8k_dates": "", "note": f"error: {exc}"})
            continue
        rows.append({
            "ticker": issuer.ticker,
            "claimed_default_date": issuer.default_date,
            "confirmed": ok,
            "item_1_03_8k_dates": ", ".join(dates),
            "note": "" if ok else "no Item 1.03 8-K in window — check the date manually",
        })
        if ok and not issuer.verified:
            updates[issuer.ticker] = {"verified": True}

    frame = pd.DataFrame(rows)
    path = _out(INTERIM) / "default_event_verification.csv"
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    if updates and not args.no_write:
        for change in update_universe_file(config_dir() / "universe.yaml", updates):
            print(f"  wrote {change}")
    unconfirmed = frame.loc[~frame["confirmed"].astype(bool), "ticker"].tolist()
    if unconfirmed:
        print(f"\nUNCONFIRMED: {unconfirmed}. These stay verified: false and block training.")
        return 1
    print("\nall default events confirmed by Item 1.03 8-Ks")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from creditrisklab.ingest.trellis_adapter import fetch_fundamentals, trellis_available

    issuers = load_issuers()
    try:
        validate_universe(issuers, require_verified=not args.allow_unverified)
    except UnresolvedUniverseError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Trellis installed: {trellis_available()}")
    out_dir = _out(INTERIM)
    frames = []
    for issuer in issuers:
        try:
            frame = fetch_fundamentals(issuer.cik, ticker=issuer.ticker, use_cache=not args.no_cache)
            frames.append(frame)
            print(f"  {issuer.ticker:6s} {len(frame):5d} facts  source={frame['source'].iloc[0]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {issuer.ticker:6s} FAILED: {exc}", file=sys.stderr)
    if not frames:
        print("no fundamentals ingested", file=sys.stderr)
        return 1
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "fundamentals.csv", index=False)
    print(f"\nwrote {out_dir/'fundamentals.csv'} ({len(combined):,} facts)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    issuers = load_issuers()
    try:
        validate_universe(issuers, require_verified=not args.allow_unverified)
    except UnresolvedUniverseError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    path = project_root() / INTERIM / "fundamentals.csv"
    if not path.exists():
        print(f"{path} not found — run `ingest` first", file=sys.stderr)
        return 1
    facts = pd.read_csv(path, dtype={"cik": str})
    facts["cik"] = facts["cik"].str.zfill(10)
    facts["period_end"] = pd.to_datetime(facts["period_end"]).dt.date
    facts["filed"] = pd.to_datetime(facts["filed"]).dt.date
    fundamentals = {cik: group for cik, group in facts.groupby("cik")}

    panel = build_panel(issuers, fundamentals)
    result = run_pipeline(panel, issuers, split_year=args.split_year, portfolio_date=_date(args.portfolio_date))

    out_dir = _out(OUTPUTS)
    panel.to_csv(out_dir / "panel.csv", index=False)
    if result.predictions is not None:
        result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    if result.exposures is not None:
        result.exposures.to_csv(out_dir / "exposures.csv", index=False)
    report.write(result, str(out_dir / "MODEL_VALIDATION.md"), load_recovery_config())
    charts.build_all(result, out_dir / "charts")
    (out_dir / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2, default=str), encoding="utf-8")

    # Committed copy for GitHub. Only the real-data command writes here; `demo` cannot, so a
    # synthetic report can never land in the published results folder.
    results_dir = _out("docs/results")
    report.write(result, str(results_dir / "MODEL_VALIDATION.md"), load_recovery_config())
    charts.build_all(result, results_dir / "charts")

    _print_summary(result)
    print(f"\nwrote {out_dir/'MODEL_VALIDATION.md'}")
    return 0


def cmd_universe(args: argparse.Namespace) -> int:
    issuers = load_issuers()
    print(json.dumps(cohort_summary(issuers), indent=2))
    meta = load_universe()["meta"]
    print(f"horizon_days={meta['horizon_days']}  window={meta['observation_window']}")
    print(f"features={len(load_model_config()['features'])}")
    return 0


def _date(value: str | None):
    if not value:
        return None
    from datetime import datetime as _dt

    return _dt.strptime(value, "%Y-%m-%d").date()


def _print_summary(result) -> None:
    d = result.diagnostics
    print(f"\nobservations={d.get('observations')}  issuers={d.get('issuers')}  positives={d.get('positive_labels')}")
    lr = result.discrimination.get("logistic_oof", {})
    z = result.discrimination.get("altman_zpp", {})
    print(f"logistic OOF AUC  = {lr.get('auc'):.3f}  95% CI [{lr.get('lower'):.3f}, {lr.get('upper'):.3f}]"
          if lr.get("auc") == lr.get("auc") else "logistic OOF AUC  = n/a")
    print(f"Altman Z'' AUC    = {z.get('auc'):.3f}" if z.get("auc") == z.get("auc") else "Altman Z'' AUC    = n/a")
    print(f"verdict           : {result.benchmark_comparison.get('verdict')}")
    cal = result.calibration
    if cal:
        print(f"mean PD  pre-correction  = {cal.get('mean_pd_before_correction', float('nan')):.2%}")
        print(f"mean PD post-correction  = {cal.get('mean_pd_after_correction', float('nan')):.2%}")
    for warning in result.warnings:
        print(f"  ! {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="creditrisklab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="offline synthetic run (watermarked)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-year", type=int, default=2021)
    p.add_argument("--portfolio-date", default="2019-06-30", help="cross-section date for EL/capital")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("resolve", help="verify CIKs against EDGAR and write verified ones to universe.yaml")
    p.add_argument("--no-write", action="store_true", help="report only; do not edit universe.yaml")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("verify-defaults", help="check bankruptcy 8-Ks against claimed default dates")
    p.add_argument("--window", type=int, default=30, help="days either side of the claimed date")
    p.add_argument("--no-write", action="store_true", help="report only; do not edit universe.yaml")
    p.set_defaults(func=cmd_verify_defaults)

    p = sub.add_parser("ingest", help="pull 10-K XBRL facts for the universe")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--allow-unverified", action="store_true", help="bypass the label verification gate (not for published results)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("run", help="full pipeline on ingested real data")
    p.add_argument("--split-year", type=int, default=2021)
    p.add_argument("--portfolio-date", default="2019-06-30",
                   help="cross-section date for EL/capital; 2019-06-30 is the observation date with the most defaults inside the horizon")
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("universe", help="print cohort summary")
    p.set_defaults(func=cmd_universe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
