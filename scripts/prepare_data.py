#!/usr/bin/env python3
"""Filter original FineDance arrays and build the matching CustomDance library."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.preprocessing.finedance import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--rest-joints", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "assets/runtime/retriever/index")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/runtime/motion_library")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/finedance_preprocessing.json")
    args = parser.parse_args()
    report = prepare(
        raw_root=args.raw_root, rest_joints=args.rest_joints, index_root=args.index,
        output_root=args.output, policy_path=args.policy,
        progress=lambda value: print(json.dumps(value), file=sys.stderr, flush=True),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
