"""Minimal console helpers with plain-print fallback (fleet-compatible).

The fleet's tests/utils/pytorch_training/console provides colored status/command/
phase helpers; this is a dependency-free stand-in so ported suite scripts import
the same names and still run on a bare node.
"""


def status(message, kind="info", stream=None):
    print(f"[{kind.upper()}] {message}", flush=True)


def command(cmd, stream=None):
    print(f"> {cmd}", flush=True)


def phase(title, kind="phase", stream=None):
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}", flush=True)
