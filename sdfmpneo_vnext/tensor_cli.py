from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import run_system_inference
from .tensor_bundle import load_tensor_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SDF-MPNEO vNext tensor-electric FAST/REFERENCE/CERTIFIED inference"
    )
    parser.add_argument(
        "bundle",
        help="tensor bundle directory published by publish_tensor_bundle",
    )
    parser.add_argument(
        "request",
        help="JSON inference request",
    )
    parser.add_argument(
        "--output",
        help="optional output JSON path; stdout is used when omitted",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device used to load FAST artifacts (default: cpu)",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    request = json.loads(
        Path(args.request).read_text(encoding="utf-8")
    )
    loaded = load_tensor_bundle(
        args.bundle,
        device=args.device,
    )
    result = run_system_inference(
        loaded.system,
        request,
    )
    text = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output:
        Path(args.output).write_text(
            text + "\n",
            encoding="utf-8",
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
