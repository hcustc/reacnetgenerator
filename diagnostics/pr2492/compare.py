"""One-runner, alternating PR #2492 measurements; no production dependencies changed."""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(value):
    """Compare deterministic outputs under the exact same dependency environment."""
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def dependency_version(name):
    """Record absent distribution metadata explicitly in a local conda smoke run."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if sys.platform == "linux":
            raise
        return None


def load_tests(root, filename):
    """Use the PR's identical benchmark definitions for every implementation."""
    spec = importlib.util.spec_from_file_location(
        filename[:-3], root / "tests" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(function):
    """Warm up, calibrate a short sample, then retain five per-call timings."""
    function()
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    loops = min(10000, max(1, int(0.08 / max(elapsed, 1e-8))))
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(loops):
            function()
        samples.append((time.perf_counter() - start) / loops)
    return {"seconds": statistics.median(samples), "samples": samples, "loops": loops}


def child(root, source, output):
    """Exercise the original benchmark closures and separately hash their results."""
    import reacnetgenerator
    from reacnetgenerator import _path, dps

    assert Path(_path.__file__).resolve().is_relative_to(source)
    tests = load_tests(root, "test_reacnetgen.py")
    results = {}
    with tempfile.TemporaryDirectory(prefix="rng-paired-") as temporary:
        os.chdir(temporary)
        for method, indices in (
            ("test_benchmark_detect", range(3)),
            ("test_benchmark_hmm", range(4)),
        ):
            for index in indices:
                param = copy.deepcopy(tests.test_data[index])
                fixture = root / "tests" / param["rngparams"]["inputfilename"]
                if fixture.is_file():
                    local = Path(temporary) / fixture.name
                    shutil.copyfile(fixture, local)
                    param["rngparams"]["inputfilename"] = str(local)
                captured = []
                getattr(tests.TestReacNetGen(), method)(captured.append, param)
                bench = captured[0]
                values = dict(
                    zip(
                        bench.__code__.co_freevars,
                        (cell.cell_contents for cell in bench.__closure__),
                        strict=True,
                    )
                )
                if method == "test_benchmark_detect":
                    result = values["detectclass"]._readstepfunc((0, values["lines"]))
                    inputs = values["lines"]
                else:
                    inputs = values["compressed_bytes"]
                    result = values["hmmclass"]._getoriginandhmm(inputs)
                row = measure(bench)
                row.update(input_sha256=digest(inputs), output_sha256=digest(result))
                results[f"{method}[{index}]"] = row
        kernels = load_tests(root, "test_step3_benchmark.py")
        for name in ("test_benchmark_atom_route", "test_benchmark_transition_graph"):
            captured = []
            getattr(kernels, name)(
                captured.append
            )  # Includes original output assertions.
            results[name] = measure(captured[0])
        cli = load_tests(root, "test_cli.py")
        for name in ("test_bench_module_import", "test_cli"):
            results[name] = measure(getattr(cli, name))
    metadata = {
        "source": str(source),
        "python": sys.version,
        "package": str(Path(reacnetgenerator.__file__).resolve()),
        "native_sha256": hashlib.sha256(Path(dps.__file__).read_bytes()).hexdigest(),
        "dependencies": {
            name: dependency_version(name)
            for name in (
                "numpy",
                "scipy",
                "networkx",
                "hmmlearn",
                "openbabel",
                "rdkit",
                "lz4",
                "pytest",
                "pytest-codspeed",
            )
        },
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


def prepare(root, sources):
    """Share only identical compiled/generated files; verify native source equality first."""
    import reacnetgenerator
    from reacnetgenerator import dps

    package = Path(reacnetgenerator.__file__).parent
    for filename in ("dps.pyx", "c_stack.cpp", "c_stack.h"):
        reference = (root / "reacnetgenerator" / filename).read_bytes()
        for source in sources.values():
            assert (source / "reacnetgenerator" / filename).read_bytes() == reference
    for source in sources.values():
        for filename in (Path(dps.__file__).name, "_version2.py"):
            shutil.copyfile(package / filename, source / "reacnetgenerator" / filename)
        bundle = Path("static/webpack/bundle.html")
        shutil.copyfile(package / bundle, source / "reacnetgenerator" / bundle)


def parent(root, sources, output, rounds):
    """Alternate implementations and include A/A controls to expose runner drift."""
    output.mkdir(parents=True, exist_ok=True)
    prepare(root, sources)
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    if affinity:
        os.sched_setaffinity(0, {affinity[0]})
    variants = {**sources, "master_control": sources["master"]}
    runs = {name: [] for name in variants}
    for repeat in range(rounds):
        order = list(variants)
        # Reverse each pair of rounds to avoid consistently favoring a later variant.
        if repeat % 2:
            order.reverse()
        for name in order:
            result = output / f"{repeat}-{name}.json"
            env = dict(
                os.environ,
                PYTHONPATH=str(variants[name]),
                PYTHONHASHSEED="0",
                PYTHONDONTWRITEBYTECODE="1",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                MPLBACKEND="Agg",
            )
            with (output / f"{repeat}-{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--child",
                        "--root",
                        str(root),
                        "--source",
                        str(variants[name]),
                        "--output",
                        str(result),
                    ],
                    cwd=output,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            runs[name].append(json.loads(result.read_text()))
            print(f"Round {repeat + 1}/{rounds}: {name} done", flush=True)
    reference = runs["head"][0]
    for group in runs.values():
        for row in group:
            for key in ("dependencies", "native_sha256", "python", "affinity"):
                assert row["metadata"][key] == reference["metadata"][key], key
            for name, values in row["results"].items():
                for key in ("input_sha256", "output_sha256"):
                    if key in values:
                        assert values[key] == reference["results"][name][key], (
                            name,
                            key,
                        )
    summary = {}
    for benchmark in reference["results"]:
        times = {
            name: [row["results"][benchmark]["seconds"] for row in rows]
            for name, rows in runs.items()
        }
        medians = {name: statistics.median(values) for name, values in times.items()}
        ratios = {
            name: [
                head / base
                for head, base in zip(times["head"], times[name], strict=True)
            ]
            for name in ("master", "merge_base")
        }
        summary[benchmark] = {
            "median_seconds": medians,
            "head_vs_master_percent": (medians["head"] / medians["master"] - 1) * 100,
            "head_vs_merge_base_percent": (medians["head"] / medians["merge_base"] - 1)
            * 100,
            "control_vs_master_percent": (
                medians["master_control"] / medians["master"] - 1
            )
            * 100,
            "paired_ratios": ratios,
            # Diagnostic threshold, not a claim of statistical significance.
            "consistent_over_10pct_vs_master": all(
                value > 1.1 for value in ratios["master"]
            ),
        }
    document = {
        "summary": summary,
        "environment": reference["metadata"],
        "platform": platform.platform(),
        "available_affinity": affinity,
        "output_equivalence": True,
        "rounds": rounds,
    }
    (output / "summary.json").write_text(json.dumps(document, indent=2))
    print(json.dumps(document, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--merge-base", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.root.resolve(), args.source.resolve(), args.output.resolve())
    else:
        parent(
            args.root.resolve(),
            {
                "master": args.master.resolve(),
                "head": args.head.resolve(),
                "merge_base": args.merge_base.resolve(),
            },
            args.output.resolve(),
            args.rounds,
        )
