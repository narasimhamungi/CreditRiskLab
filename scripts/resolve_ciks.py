"""Thin wrapper: equivalent to `python -m creditrisklab.cli resolve`. Extra arguments pass through."""

import sys

from creditrisklab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["resolve", *sys.argv[1:]]))
