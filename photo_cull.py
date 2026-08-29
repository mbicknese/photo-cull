#!/usr/bin/env python3
"""Thin entry-point script so the tool can be invoked as `python photo_cull.py ...`.

The real implementation lives in the `photo_cull` package.
"""
from photo_cull.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
