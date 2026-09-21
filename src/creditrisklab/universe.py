"""Issuer and default-event domain objects.

The pipeline refuses to consume an issuer whose CIK is unresolved or whose default event
is unverified. That is deliberate: the entire project rests on the labels being real, so
an unverified label is treated as a hard error, not a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

from creditrisklab.config import load_universe


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


@dataclass(frozen=True)
class Issuer:
    name: str
    ticker: str
    sector: str
    cik: str | None = None
    edgar_name_query: str | None = None
    default_date: date | None = None
    default_type: str | None = None
    verified: bool = False
    exposure: dict[str, float] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return self.default_date is not None

    def validate(self, require_verified: bool = True) -> None:
        if not self.cik:
            raise UnresolvedUniverseError(
                f"{self.ticker}: CIK unresolved. Run scripts/resolve_ciks.py before the pipeline."
            )
        if self.is_default:
            if not self.exposure:
                raise UnresolvedUniverseError(f"{self.ticker}: defaulted issuer has no exposure mix for LGD/EAD")
            total = sum(self.exposure.values())
            if abs(total - 1.0) > 1e-6:
                raise UnresolvedUniverseError(f"{self.ticker}: exposure weights sum to {total:.4f}, expected 1.0")
            if require_verified and not self.verified:
                raise UnresolvedUniverseError(
                    f"{self.ticker}: default event dated {self.default_date} is unverified. "
                    "Run scripts/verify_default_events.py; unverified labels are not usable ground truth."
                )


class UnresolvedUniverseError(RuntimeError):
    """Raised when the universe is not yet fit to train on."""


def load_issuers(path: str | None = None) -> list[Issuer]:
    cfg = load_universe(path)
    issuers: list[Issuer] = []
    for row in cfg["defaults"]:
        issuers.append(
            Issuer(
                name=row["name"],
                ticker=row["ticker"],
                sector=row.get("sector", "unknown"),
                cik=_norm_cik(row.get("cik")),
                edgar_name_query=row.get("edgar_name_query"),
                default_date=_parse_date(row.get("default_date")),
                default_type=row.get("default_type"),
                verified=bool(row.get("verified", False)),
                exposure=dict(row.get("exposure", {})),
            )
        )
    for row in cfg["non_defaults"]:
        issuers.append(
            Issuer(
                name=row["name"],
                ticker=row["ticker"],
                sector=row.get("sector", "unknown"),
                cik=_norm_cik(row.get("cik")),
                edgar_name_query=row.get("edgar_name_query"),
                verified=True,  # absence of a default event needs no event verification
            )
        )
    return issuers


def _norm_cik(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value).lstrip("0").zfill(10)


def validate_universe(issuers: Iterable[Issuer], require_verified: bool = True) -> None:
    problems: list[str] = []
    for issuer in issuers:
        try:
            issuer.validate(require_verified=require_verified)
        except UnresolvedUniverseError as exc:
            problems.append(str(exc))
    if problems:
        raise UnresolvedUniverseError(
            "universe is not ready:\n  - " + "\n  - ".join(problems)
        )


def cohort_summary(issuers: Iterable[Issuer]) -> dict[str, int]:
    issuers = list(issuers)
    return {
        "issuers": len(issuers),
        "defaults": sum(1 for i in issuers if i.is_default),
        "non_defaults": sum(1 for i in issuers if not i.is_default),
        "verified_defaults": sum(1 for i in issuers if i.is_default and i.verified),
    }


# --------------------------------------------------------------------------------------
# Writing resolved values back to universe.yaml
# --------------------------------------------------------------------------------------
# Text-level edits, not a YAML round-trip: a safe_load/safe_dump cycle would strip every
# comment in the file, and the comments carry the fact-discipline notes. CIKs are always
# written as quoted strings — unquoted, YAML 1.1 reads a zero-padded CIK made only of the
# digits 0-7 (Frontier's 0000020520) as an octal integer, silently producing a wrong CIK.

import re as _re
from pathlib import Path as _Path


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f'"{value}"'


def update_universe_file(path: str | _Path, updates: dict[str, dict[str, object]]) -> list[str]:
    """Set fields for tickers in universe.yaml. `updates` = {ticker: {field: value}}.

    Handles both layouts used in the file: block entries (defaults) and one-line flow
    mappings (non_defaults). Returns a human-readable list of the changes made.
    """
    path = _Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changes: list[str] = []

    for ticker, fields in updates.items():
        tick = _re.escape(ticker)
        flow = _re.compile(rf"\bticker:\s*{tick}\s*,")
        block = _re.compile(rf"^\s*ticker:\s*{tick}\s*(#.*)?$")
        for i, line in enumerate(lines):
            if flow.search(line):
                for field, value in fields.items():
                    new, n = _re.subn(rf"(\b{field}:\s*)([^,}}]+)", lambda m: m.group(1) + _format_value(value), line, count=1)
                    if n:
                        line = new
                        changes.append(f"{ticker}.{field} = {_format_value(value)}")
                lines[i] = line
                break
            if block.match(line.rstrip("\n")):
                j = i + 1
                while j < len(lines) and not _re.match(r"^\s*-\s+name:|^\S", lines[j]):
                    for field, value in fields.items():
                        m = _re.match(rf"^(\s*{field}:\s*)([^#\n]*?)(\s*#.*)?(\n?)$", lines[j])
                        if m:
                            lines[j] = f"{m.group(1)}{_format_value(value)}{m.group(3) or ''}{m.group(4)}"
                            changes.append(f"{ticker}.{field} = {_format_value(value)}")
                    j += 1
                # the ticker line sits after `name:`; fields such as cik follow it
                break

    path.write_text("".join(lines), encoding="utf-8")
    load_universe.cache_clear()
    return changes


def default_event_confirmed(filings, default_date, window_days: int = 30) -> tuple[bool, list[str]]:
    """True if an 8-K reporting Item 1.03 (bankruptcy or receivership) was filed within
    `window_days` of the claimed default date. Returns (confirmed, matching filing dates)."""
    import pandas as pd
    from datetime import timedelta

    if filings is None or len(filings) == 0 or default_date is None:
        return False, []
    frame = filings.copy()
    frame["filingDate"] = pd.to_datetime(frame["filingDate"], errors="coerce").dt.date
    lo, hi = default_date - timedelta(days=window_days), default_date + timedelta(days=window_days)
    near = frame[(frame["filingDate"] >= lo) & (frame["filingDate"] <= hi)]
    if "items" not in near.columns:
        return False, []
    hits = near[near["items"].astype(str).str.contains(r"\b1\.03\b", regex=True)]
    return (not hits.empty), sorted(str(d) for d in hits["filingDate"])
