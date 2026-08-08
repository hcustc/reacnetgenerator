# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line validation for ReacNetGenerator timed-output HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ._timedoutputvalidate import (
    build_timed_output_manifest,
    compare_timed_output_manifests,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reacnetgenerator-check-timed-output",
        description=(
            "Validate a completed timed-output HDF5 file and create a "
            "bounded-memory semantic manifest."
        ),
    )
    parser.add_argument("filename", type=Path, help="timed-output HDF5 file")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="JSON manifest whose semantic content must match",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the candidate JSON manifest here instead of stdout",
    )
    parser.add_argument(
        "--require-same-source-paths",
        action="store_true",
        help="also require the stored source paths to match the baseline",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=4096,
        help="maximum rows per metadata/event validation read (default: 4096)",
    )
    parser.add_argument(
        "--block-mib",
        type=int,
        default=16,
        help="maximum molecule payload bytes per read in MiB (default: 16)",
    )
    return parser


def _write_json_atomic(path: Path, payload: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    """Validate one file, optionally compare it with a baseline manifest."""
    arguments = _parser().parse_args(argv)
    try:
        if (
            arguments.output is not None
            and arguments.output.resolve() == arguments.filename.resolve()
        ):
            raise ValueError("--output must not overwrite the input HDF5 file")
        if (
            arguments.output is not None
            and arguments.baseline is not None
            and arguments.output.resolve() == arguments.baseline.resolve()
        ):
            raise ValueError("--output must not overwrite the baseline manifest")
        manifest = build_timed_output_manifest(
            arguments.filename,
            block_rows=arguments.block_rows,
            block_bytes=arguments.block_mib * 1024**2,
        )
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if arguments.output is None:
            sys.stdout.write(encoded)
        else:
            _write_json_atomic(arguments.output, encoded)
        if arguments.baseline is not None:
            with arguments.baseline.open(encoding="utf-8") as stream:
                baseline = json.load(stream)
            mismatches = compare_timed_output_manifests(
                manifest,
                baseline,
                include_provenance=arguments.require_same_source_paths,
            )
            if mismatches:
                print("Timed-output semantic comparison failed:", file=sys.stderr)
                for mismatch in mismatches:
                    print(f"- {mismatch}", file=sys.stderr)
                return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Timed-output validation failed: {error}", file=sys.stderr)
        return 2
    return 0


def _commandline() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - exercised through entry point
    _commandline()
