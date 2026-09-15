"""Compare bounded route/reaction workloads before and after the scheduling fix."""

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from compare import prepare


def synthetic(directory, case):
    """Time actual route and reaction dispatch, including pool/mapping overhead."""
    from reacnetgenerator._path import _CollectSMILESPaths
    from reacnetgenerator._reaction import ReactionsFinder
    from reacnetgenerator._step3state import _AtomFrameStore, _MoleculeNameTable

    atoms, frames, stride, nproc, events = {
        "short": (16, 32, 8, 2, False),
        "larger_dense_counts": (512, 256, 1, 2, False),
        "larger_sparse_counts": (512, 256, 64, 2, False),
        "sparse_counts": (16, 2048, 256, 2, False),
        "dense_counts": (16, 2048, 1, 2, False),
        "dense_events": (16, 2048, 1, 2, True),
        "dense_serial": (16, 2048, 1, 1, False),
    }[case]
    names = _MoleculeNameTable.from_names(["A", "B"])
    collector = object.__new__(_CollectSMILESPaths)
    collector.N, collector.step, collector.nproc = atoms, frames, nproc
    collector.atomname, collector.atomtype = np.array(["H"]), np.zeros(atoms, dtype=int)
    collector.selectatoms, collector.mname, collector.runHMM = ["H"], names, True
    collector.atomroutefilename = str(directory / "out.route")
    finder = ReactionsFinder(
        SimpleNamespace(
            step=frames,
            mname=names,
            nproc=nproc,
            printreactionevent=events,
            reactionabcdfilename=str(directory / "out.reactionabcd"),
            reactioneventfilename=str(directory / "out.events"),
        )
    )
    with _AtomFrameStore((atoms, frames), 2, directory=directory) as store:
        store.atomeach[:] = np.resize(np.repeat([1, 2], stride), frames)
        store.flush()
        indexed = hasattr(store, "prepare_active_transitions")
        cpu_start = sum(
            getattr(resource.getrusage(who), field)
            for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
            for field in ("ru_utime", "ru_stime")
        )
        start = time.perf_counter()
        if indexed:
            store.prepare_active_transitions()
            collector._printatomroute(store)
        else:
            collector._printatomroute(store.atomeach)
        route_seconds = time.perf_counter() - start
        start = time.perf_counter()
        finder.findreactions(
            store.atomeach.T,
            store.conflict.T,
            **({"matrix_store": store} if indexed else {}),
        )
        reaction_seconds = time.perf_counter() - start
        cpu_seconds = (
            sum(
                getattr(resource.getrusage(who), field)
                for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
                for field in ("ru_utime", "ru_stime")
            )
            - cpu_start
        )
        disk_bytes = store.nbytes
        tasks = store.active_transition_count if indexed else frames - 1
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("out.*")
        if p.is_file()
    }
    return {
        "indexed": indexed,
        "shape": [atoms, frames],
        "route_seconds": route_seconds,
        "reaction_seconds": reaction_seconds,
        "seconds": route_seconds + reaction_seconds,
        "cpu_seconds": cpu_seconds,
        "matrix_disk_bytes": disk_bytes,
        "transitions_considered": tasks,
        "files": files,
    }


def run_comparison(sources, output):
    """Alternate all three versions and reject any changed scientific output."""
    prepare(sources["fixed"], sources)
    output.mkdir(parents=True, exist_ok=True)
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(available[:2]))
    cases = (
        "short",
        "sparse_counts",
        "dense_counts",
        "dense_events",
        "dense_serial",
        "larger_dense_counts",
        "larger_sparse_counts",
    )
    runs = {}
    for case in cases:
        runs[case] = {name: [] for name in sources}
        for repeat in range(5):
            order = list(sources)
            if repeat % 2:
                order.reverse()
            pair = {}
            for name in order:
                target = output / case / str(repeat) / name
                target.mkdir(parents=True, exist_ok=True)
                env = dict(
                    os.environ,
                    PYTHONPATH=str(sources[name]),
                    PYTHONHASHSEED="0",
                    PYTHONDONTWRITEBYTECODE="1",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    MPLBACKEND="Agg",
                )
                with (target / "run.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--child",
                            case,
                            "--source",
                            str(sources[name]),
                            "--output",
                            str(target),
                        ],
                        cwd=output,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                    )
                result = json.loads((target / "result.json").read_text())
                runs[case][name].append(result)
                pair[name] = result
                print(case, repeat, name, round(result["seconds"], 5), flush=True)
            assert (
                pair["base"]["files"]
                == pair["before"]["files"]
                == pair["fixed"]["files"]
            ), case
            assert (
                pair["base"]["native_sha256"]
                == pair["before"]["native_sha256"]
                == pair["fixed"]["native_sha256"]
            )
    summary = {}
    for case, variants in runs.items():
        medians = {
            name: {
                field: statistics.median(row[field] for row in rows)
                for field in (
                    "seconds",
                    "route_seconds",
                    "reaction_seconds",
                    "cpu_seconds",
                )
            }
            for name, rows in variants.items()
        }
        summary[case] = {
            "medians": medians,
            "before_vs_base_percent": 100
            * (medians["before"]["seconds"] / medians["base"]["seconds"] - 1),
            "fixed_vs_base_percent": 100
            * (medians["fixed"]["seconds"] / medians["base"]["seconds"] - 1),
            "fixed_vs_before_percent": 100
            * (medians["fixed"]["seconds"] / medians["before"]["seconds"] - 1),
            "fixed_base_paired_ratios": [
                a["seconds"] / b["seconds"]
                for a, b in zip(variants["fixed"], variants["base"], strict=True)
            ],
        }
    (output / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "output_equivalence": True,
                "rounds": 5,
                "cpu_affinity": available[:2],
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--fixed", type=Path)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        import reacnetgenerator
        from reacnetgenerator import _path, dps

        assert Path(_path.__file__).resolve().is_relative_to(args.source.resolve())
        args.output.mkdir(parents=True, exist_ok=True)
        result = synthetic(args.output.resolve(), args.child)
        result.update(
            source=str(Path(reacnetgenerator.__file__).resolve()),
            native_sha256=hashlib.sha256(Path(dps.__file__).read_bytes()).hexdigest(),
            path_sha256=hashlib.sha256(Path(_path.__file__).read_bytes()).hexdigest(),
            python=sys.version,
            numpy=np.__version__,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    else:
        run_comparison(
            {
                "base": args.base.resolve(),
                "before": args.before.resolve(),
                "fixed": args.fixed.resolve(),
            },
            args.output.resolve(),
        )
