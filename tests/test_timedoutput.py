# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized, bounded-memory timed output."""

import os
import shutil
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

from reacnetgenerator import ReacNetGenerator
from reacnetgenerator._detect import _Detect
from reacnetgenerator._path import (
    _AtomFrameStore,
    _CollectPaths,
    _CollectSMILESPaths,
    _get_atom_route_by_index,
    _initialize_route_worker,
)
from reacnetgenerator._reaction import (
    _get_transition_reactions_by_index,
    _initialize_reaction_worker,
)
from reacnetgenerator._timedoutput import TimedOutputStore
from reacnetgenerator.commandline import main_parser, parm2cmd
from reacnetgenerator.tools import (
    iter_molecule_timeline,
    iter_reaction_events,
    read_timed_output_metadata,
)
from reacnetgenerator.utils import bytestolist, listtobytes


def _new_store(
    output,
    *,
    timestep=None,
    frame_source=None,
    molecule_enabled=True,
    reaction_enabled=True,
):
    if timestep is None:
        timestep = {0: 100, 1: 200, 2: 100, 3: 300}
    if frame_source is None:
        frame_source = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1)}
    return TimedOutputStore(
        str(output),
        cache_mib=4,
        input_filenames=["first.dump", "second.dump"],
        timestep=timestep,
        frame_source=frame_source,
        stepinterval=2,
        molecule_enabled=molecule_enabled,
        reaction_enabled=reaction_enabled,
    )


def _decode(values):
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def test_timed_output_schema_ranges_sources_and_reactions(tmp_path):
    """Persist HDF5 ranges, provenance, and compact reaction counters."""
    output = tmp_path / "trajectory.timeline.h5"
    store = _new_store(output)
    with store:
        store.add_molecule(1, "A", [0, 1], [[0, 1, 1]], [(0, 2)])
        store.add_molecule(2, "B", [2], [], [(3, 3)])
        store.stage_reaction_events(1, Counter({("A", "B"): 2, ("B", "C"): 1}))
        store.stage_reaction_events(0, Counter({("B", "C"): 3}))
        store.finalize_reactions()
        store.finalize_and_publish()

    assert output.exists()
    assert not list(tmp_path.glob("trajectory.timeline.h5.tmp.*"))
    assert not (tmp_path / "trajectory.timeline.h5.lock").exists()
    assert read_timed_output_metadata(output)["status"] == "complete"

    with h5py.File(output, "r") as handle:
        assert set(handle) == {
            "frames",
            "molecule_ranges",
            "molecules",
            "reaction_events",
            "reaction_types",
            "sources",
            "species",
        }
        assert _decode(handle["sources/path"][:]) == ["first.dump", "second.dump"]
        np.testing.assert_array_equal(handle["sources/ordinal"][:], [0, 1])
        np.testing.assert_array_equal(handle["frames/source_id"][:], [1, 1, 2, 2])
        np.testing.assert_array_equal(handle["frames/source_frame"][:], [0, 1, 0, 1])
        np.testing.assert_array_equal(
            handle["frames/timestep"][:], [100, 200, 100, 300]
        )
        np.testing.assert_array_equal(handle["molecule_ranges/molecule_id"][:], [1, 2])
        np.testing.assert_array_equal(handle["molecule_ranges/start_frame"][:], [0, 3])
        np.testing.assert_array_equal(handle["molecule_ranges/end_frame"][:], [2, 3])
        assert _decode(handle["reaction_types/reactant"][:]) == ["A", "B"]
        assert _decode(handle["reaction_types/product"][:]) == ["B", "C"]
        np.testing.assert_array_equal(handle["reaction_types/total_count"][:], [2, 4])
        np.testing.assert_array_equal(
            handle["reaction_events/transition_index"][:], [1, 1, 0]
        )
        np.testing.assert_array_equal(
            handle["reaction_events/reaction_id"][:], [1, 2, 2]
        )
        np.testing.assert_array_equal(handle["reaction_events/count"][:], [2, 1, 3])
        np.testing.assert_array_equal(
            handle["reaction_events/block_start"][:], [2, 0, 0]
        )
        np.testing.assert_array_equal(
            handle["reaction_events/block_length"][:], [1, 2, 0]
        )

    assert list(iter_molecule_timeline(output)) == [
        (100, "A", "0;1", "0-1-1"),
        (200, "A", "0;1", "0-1-1"),
        (100, "A", "0;1", "0-1-1"),
        (300, "B", "2", ""),
    ]
    assert list(iter_reaction_events(output)) == [
        (0, "B", "C"),
        (0, "B", "C"),
        (0, "B", "C"),
        (1, "A", "B"),
        (1, "A", "B"),
        (1, "B", "C"),
    ]


def test_invalid_transition_keeps_failed_temporary_hdf5(tmp_path):
    """Reject out-of-range transitions without replacing the formal file."""
    output = tmp_path / "invalid.timeline.h5"
    store = _new_store(
        output,
        timestep={0: 100},
        frame_source={0: (1, 0)},
        molecule_enabled=False,
        reaction_enabled=True,
    )
    with pytest.raises(RuntimeError, match="transition_index"):
        with store:
            store.stage_reaction_events(0, Counter({("A", "B"): 1}))

    assert not output.exists()
    assert store.status == "failed"
    temporary = tmp_path / f"invalid.timeline.h5.tmp.{store.job_id}"
    with h5py.File(temporary, "r") as handle:
        assert handle.attrs["status"] == "failed"


def test_target_lock_prevents_concurrent_build_and_cleanup(tmp_path):
    """Protect an active HDF5 build from another writer and orphan cleanup."""
    output = tmp_path / "locked.timeline.h5"
    first = _new_store(output, molecule_enabled=False, reaction_enabled=False)
    second = _new_store(output, molecule_enabled=False, reaction_enabled=False)
    with first:
        TimedOutputStore.cleanup_orphans(str(output), retention_seconds=-1)
        assert Path(first.temp_filename).exists()
        with pytest.raises(RuntimeError, match="already being built"):
            second.__enter__()


def test_stale_orphan_cleanup_uses_file_age(tmp_path):
    """Remove only expired job-specific HDF5 temporary files."""
    output = tmp_path / "orphan.timeline.h5"
    stale = tmp_path / "orphan.timeline.h5.tmp.deadjob"
    recent = tmp_path / "orphan.timeline.h5.tmp.recentjob"
    stale.write_bytes(b"diagnostic")
    recent.write_bytes(b"diagnostic")
    old = time.time() - 100
    os.utime(stale, (old, old))

    TimedOutputStore.cleanup_orphans(str(output), retention_seconds=10)

    assert not stale.exists()
    assert recent.exists()


def test_persistent_molecule_output_is_stored_as_ranges(tmp_path):
    """Represent one million logical occurrences with only 100 range rows."""
    output = tmp_path / "persistent.timeline.h5"
    frame_count = 10_000
    molecule_count = 100
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(frame_count)},
        frame_source={frame: (1, frame) for frame in range(frame_count)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    with store:
        for molecule_id in range(1, molecule_count + 1):
            store.add_molecule(
                molecule_id,
                "A",
                [molecule_id],
                [],
                [(0, frame_count - 1)],
            )
        store.finalize_and_publish()

    with h5py.File(output, "r") as handle:
        ranges = handle["molecule_ranges"]
        assert len(ranges["molecule_id"]) == molecule_count
        logical_rows = np.sum(ranges["end_frame"][:] - ranges["start_frame"][:] + 1)
        assert int(logical_rows) == frame_count * molecule_count


def test_legacy_csv_filename_parameters_have_migration_error(tmp_path):
    """Report the HDF5 replacement for removed CSV filename parameters."""
    with pytest.raises(TypeError, match="timedoutputfilename"):
        ReacNetGenerator(
            inputfilename=str(tmp_path / "dummy"),
            inputfiletype="lammpsbondfile",
            atomname=["H"],
            moleculetimelinefilename="legacy.csv",
        )


def test_detect_tracks_multiple_source_files(tmp_path):
    """Apply stepinterval globally while preserving selected frame provenance."""
    source = Path(__file__).parent / "inputs" / "water.bond"
    files = [tmp_path / f"source-{index}.bond" for index in range(3)]
    for filename in files:
        shutil.copyfile(source, filename)
    rng = ReacNetGenerator(
        inputfilename=[str(filename) for filename in files],
        inputfiletype="bond",
        atomname=["H", "O"],
        nproc=1,
        runHMM=False,
        stepinterval=2,
    )

    _Detect.gettype(rng).detect()

    assert rng.step == 2
    assert rng.framesource == {0: (1, 0), 1: (3, 0)}
    assert rng.timestep[0] == rng.timestep[1]


@pytest.mark.parametrize(
    ("molecule_count", "itemsize"),
    [
        (255, 1),
        (256, 2),
        (2**16 - 1, 2),
        (2**16, 4),
        (2**32 - 1, 4),
    ],
)
def test_path_uses_smallest_unsigned_molecule_dtype(molecule_count, itemsize):
    """Keep the two PATH matrices within five bytes per atom-frame cell."""
    dtype = _CollectPaths._molecule_index_dtype(molecule_count)

    assert np.issubdtype(dtype, np.unsignedinteger)
    assert dtype.itemsize == itemsize
    assert dtype.itemsize + np.dtype(np.bool_).itemsize <= 5


def test_smiles_worker_returns_compressed_frame_block(tmp_path):
    """Keep frame timelines compressed while crossing worker IPC."""
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        printmoleculetime=True,
    )
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])
    frame_block = listtobytes(np.arange(1000, dtype=np.uint64))
    line = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
        frame_block,
    )

    _, _, _, returned_frames = collector._calmoleculeSMILESname(line)

    assert returned_frames is frame_block
    assert isinstance(returned_frames, bytes)
    np.testing.assert_array_equal(
        bytestolist(returned_frames), np.arange(1000, dtype=np.uint64)
    )


def test_step3_workers_receive_only_indices_and_map_shared_files():
    """Attach shared mappings once and send integer atom/transition indices."""
    store = _AtomFrameStore((2, 4), np.uint8)
    try:
        store.atomeach[:] = [[1, 1, 2, 2], [3, 3, 3, 3]]
        store.conflict[:] = False
        store.save_molecule_names(np.array(["A", "B", "C"]))
        store.save_atom_types(np.array([0, 0]))
        store.flush()

        _initialize_route_worker(
            store.atomeach_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
            store.atomtype_path,
            np.array(["H"]),
            ("H",),
            (0, 4),
        )
        route, route_text = _get_atom_route_by_index(0)
        np.testing.assert_array_equal(route, [[1, 2]])
        assert route_text == "Atom 1 H: 0 A -> 2 B"

        store.atomeach[:, :2] = [[1, 3], [2, 3]]
        store.flush()
        _initialize_reaction_worker(
            store.atomeach_path,
            store.conflict_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
        )
        transition_index, counter = _get_transition_reactions_by_index(0)
        assert transition_index == 0
        assert counter == Counter({("A+B", "C"): 1})
    finally:
        store.close()


def test_timed_output_cli_and_command_roundtrip():
    """Expose the HDF5 output filename and chunk-cache size through the CLI."""
    args = main_parser().parse_args(
        [
            "-i",
            "input.bond",
            "-a",
            "H",
            "--show-molecule-time",
            "--reaction-event",
            "--timed-output",
            "result.h5",
            "--timed-output-cache-mib",
            "32",
        ]
    )
    command = parm2cmd(
        {
            "inputfilename": "input.bond",
            "inputfiletype": "bond",
            "atomname": ["H"],
            "printmoleculetime": True,
            "printreactionevent": True,
            "timedoutputfilename": "result.h5",
            "timedoutputcachemib": 32,
        }
    )

    assert args.timedoutputfilename == "result.h5"
    assert args.timedoutputcachemib == 32
    assert command[command.index("--timed-output") + 1] == "result.h5"
    assert command[command.index("--timed-output-cache-mib") + 1] == "32"


def test_end_to_end_timed_output(tmp_path):
    """Publish a complete HDF5 file from a real bond trajectory."""
    source = Path(__file__).parent / "inputs" / "water.bond"
    trajectory = tmp_path / "water.bond"
    shutil.copyfile(source, trajectory)
    output = tmp_path / "water.timeline.h5"
    rng = ReacNetGenerator(
        inputfilename=str(trajectory),
        inputfiletype="bond",
        atomname=["H", "O"],
        nproc=1,
        runHMM=False,
        needprintspecies=False,
        printmoleculetime=True,
        printreactionevent=True,
        timedoutputfilename=str(output),
    )

    rng.run()

    with h5py.File(output, "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert len(handle["frames/timestep"]) == 1
        assert len(handle["molecules/molecule_id"]) == 1
        np.testing.assert_array_equal(handle["molecule_ranges/start_frame"][:], [0])
        np.testing.assert_array_equal(handle["molecule_ranges/end_frame"][:], [0])
        assert len(handle["reaction_events/count"]) == 0
