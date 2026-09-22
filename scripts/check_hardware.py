#!/usr/bin/env python3
"""Report what this machine offers the DL branch; run first on the DGX Spark.

    python scripts/check_hardware.py                 # detect and report
    python scripts/check_hardware.py --smoke         # + tiny bf16 matmul / ESM forward
    python scripts/check_hardware.py --out runs/dl/hardware.json

Detection only by default — no benchmark, no download. ``--smoke`` proves the
stack runs on the GPU; it measures nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpdl.dl.hardware import hardware_report  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = hardware_report(smoke=args.smoke)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    for note in report["recommendations"]:
        print(f"NOTE: {note}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
