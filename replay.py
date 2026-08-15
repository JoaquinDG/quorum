#!/usr/bin/env python3
"""Reconstruct a Quorum session from its trace file.

    python3 replay.py traces/quickstart.jsonl

The work lives in `quorum.replay`; this is the front door. It exists because
the trace format's central claim — that any renderer is a player for the file
— is worth being able to check in one command, without knowing anything about
the package layout.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from quorum.replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
