"""`stabilis validate` -- run the evidence report and return its verdict.

The exit status is the verdict, so this works as a gate in a lab's own
qualification pipeline rather than only as something to read.
"""

from __future__ import annotations

import sys

__all__ = ["add_validate"]


def add_validate(sub) -> None:
    p_val = sub.add_parser("validate", help="run the validation evidence report")
    p_val.add_argument("--out", help="write an HTML report to this path")
    p_val.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p_val.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Monte-Carlo depth; raised to 300 if lower (default: 300, "
        "published figures use 800)",
    )
    p_val.add_argument(
        "--self-test",
        action="store_true",
        help="inject a check designed to fail, proving the report can",
    )
    p_val.set_defaults(func=_cmd_validate)


def _cmd_validate(args) -> int:
    """Run the validation hierarchy and emit the evidence report.

    Exit status is the verdict: 0 if every check passed, 1 otherwise. That
    makes the report usable as a gate in a lab's own qualification pipeline
    rather than only as something to read.
    """
    from ..validate import run, to_html, to_json

    report = run(trials=args.trials, self_test=args.self_test)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(to_html(report))
    if args.json:
        print(to_json(report))
    else:
        for check in report.checks:
            mark = "pass" if check.passed else "FAIL"
            print(f"[{mark}] {check.tier:<10} {check.name}")
            print(f"         {check.measured}")
        passed = sum(c.passed for c in report.checks)
        print(
            f"\n{'PASS' if report.passed else 'FAIL'}: {passed}/"
            f"{len(report.checks)} checks   digest {report.digest[:16]}"
        )
        if args.out:
            print(f"report written to {args.out}")
    return 0 if report.passed else 1
