"""Thin wrapper: equivalent to `python -m creditrisklab.cli verify-defaults`. Extra arguments pass through."""

import sys

from creditrisklab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["verify-defaults", *sys.argv[1:]]))
