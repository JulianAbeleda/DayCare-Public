"""CLI for generic image-style training projects."""

from __future__ import annotations

import argparse
from pathlib import Path

from .project import ImageProject


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m daycare.image")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a generic image-style project")
    init.add_argument("path", type=Path)
    init.add_argument("--name", required=True)
    init.add_argument("--trigger", required=True)
    init.add_argument("--base-model", required=True)

    validate = subparsers.add_parser("validate", help="validate dataset and captions")
    validate.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "init":
        project = ImageProject.create(
            args.path,
            name=args.name,
            trigger=args.trigger,
            base_model=args.base_model,
        )
        print(project.root)
        return 0

    report = ImageProject.load(args.path).validate()
    for error in report.errors:
        print(f"ERROR: {error}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    print(f"train={report.train_images} holdout={report.holdout_images}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
