"""List Trellis-validated issuers missing from the CreditRiskLab non-default cohort.

Read-only by design. It prints what to add; it does not edit config/universe.yaml. A
universe file that rewrites itself is a universe file nobody can audit.
"""

from creditrisklab.config import load_universe
from creditrisklab.ingest.trellis_adapter import trellis_available, trellis_universe_tickers


def main() -> int:
    if not trellis_available():
        print("Trellis is not installed in this environment (pip install -e path/to/trellis).")
        return 1
    trellis = set(trellis_universe_tickers())
    if not trellis:
        print("Trellis is installed but exposes no universe attribute this script recognises. "
              "List its validated tickers manually from the Trellis README.")
        return 1
    cfg = load_universe()
    present = {row["ticker"].upper() for row in cfg["non_defaults"]} | {row["ticker"].upper() for row in cfg["defaults"]}
    missing = sorted(trellis - present)
    print(f"Trellis universe: {len(trellis)}   already in CreditRiskLab: {len(trellis & present)}")
    for ticker in missing:
        print(f'  - {{ name: "TODO", ticker: {ticker}, edgar_name_query: "TODO", cik: null, sector: "TODO" }}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
