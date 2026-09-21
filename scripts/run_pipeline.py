"""Thin wrapper: equivalent to `python -m creditrisklab.cli run`. Extra arguments pass through."""

import sys

from creditrisklab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
