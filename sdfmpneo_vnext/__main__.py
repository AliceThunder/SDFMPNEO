from __future__ import annotations

import sys

from .cli import main as _cli_main


def main(
    argv=None,
) -> int:
    if argv is None:
        argv = sys.argv[
            1:
        ]
    else:
        argv = list(
            argv
        )
    if argv == [
        "--self-check"
    ]:
        argv = [
            "self-check"
        ]
    return _cli_main(
        argv
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
