#!/usr/bin/env python3
"""
upload_from_artifacts.py — build the RASTRA payload from a results dir and
(dry-run) write logs/payload.json. Thin wrapper over
tests/utils/results/handler.upload_results, mirroring the vLLM/xDiT uploader.

STUB: no network POST — the handler is dry-run by design until creds are wired.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="logs/benchmark_results")
    p.add_argument("--execution-label", default="")
    p.add_argument("--save-payload", default="logs/payload.json")
    args = p.parse_args()

    try:
        from tests.utils.results.handler import upload_results
    except Exception as e:
        print(f"[warn] results handler unavailable: {e}; skipping upload.")
        return 0

    upload_results(args.results_dir, execution_label=args.execution_label,
                   out_path=args.save_payload, dry_run=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
