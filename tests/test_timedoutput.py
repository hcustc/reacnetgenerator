# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for normalized, bounded-memory timed output."""

import builtins
import os
import shutil
import time
import weakref
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import reacnetgenerator._hmmfilter as hmmfilter_module
import reacnetgenerator._matrix as matrix_module
import reacnetgenerator._packedbool as packedbool_module
import reacnetgenerator._path as path_module
import reacnetgenerator._reaction as reaction_module
import reacnetgenerator._timedoutput as timedoutput_module
import reacnetgenerator._timedoutputvalidate as timedoutputvalidate_module
import reacnetgenerator.utils as utils_module
from reacnetgenerator import ReacNetGenerator
from reacnetgenerator._detect import _Detect
from reacnetgenerator._hmmfilter import _HMMFilter
from reacnetgenerator._matrix import _GenerateMatrix
from reacnetgenerator._path import (
    _AtomFrameStore,
    _calculate_atom_route,
    _CollectPaths,
    _CollectSMILESPaths,
    _get_atom_route_by_index,
    _initialize_route_worker,
    _iter_molecule_fields,
)
from reacnetgenerator._reaction import (
    _calculate_transition_reactions,
    _get_transition_reactions_by_index,
    _initialize_reaction_worker,
)
from reacnetgenerator._timedoutput import TimedOutputStore
from reacnetgenerator.commandline import main_parser, parm2cmd
from reacnetgenerator.timedoutputcheck import main as timed_output_check_main
from reacnetgenerator.tools import (
    build_timed_output_manifest,
    compare_timed_output_manifests,
    iter_molecule_timeline,
    iter_reaction_events,
    read_timed_output_metadata,
)
from reacnetgenerator.utils import (
    WriteBuffer,
    _DiskOrderedResultSpool,
    listtobytes,
    run_mp,
)


def _head_of_line_task(index):
    started = time.monotonic()
    time.sleep(0.3 if index == 0 else 0.01)
    return index, started, time.monotonic()


def _failing_timed_task(index):
    if index == 3:
        raise RuntimeError("intentional ordered worker failure")
    return index


def _slow_tail_task(index):
    if index:
        time.sleep(1)
    return index


class _DelayedNoHMMFilter(_HMMFilter):
    """Force later dense records to finish before an earlier record."""

    def _getoriginandhmm(self, item):
        atom = int(np.asarray(utils_module.bytestolist(item[0])).reshape((-1,))[0])
        if atom == 1:
            time.sleep(0.3)
        elif atom == 2:
            time.sleep(0.01)
        return super()._getoriginandhmm(item)


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


def test_write_buffer_flushes_before_byte_limit(tmp_path):
    """Bound retained route-sized payloads by bytes instead of row count alone."""
    output = tmp_path / "bounded-buffer.bin"
    with open(output, "wb") as handle:
        buffer = WriteBuffer(
            handle,
            linenumber=1000,
            sep=b"\n",
            byte_limit=8,
        )
        buffer.append(b"12345")
        assert buffer.buff == [b"12345"]

        buffer.append(b"67890")

        assert buffer.buff == [b"67890"]
        assert handle.tell() == len(b"12345\n")

        buffer.append(b"123456789")

        assert buffer.buff == []
        assert buffer.maximum_buffer_bytes == len(b"123456789\n")
        buffer.flush()

    assert output.read_bytes() == b"12345\n67890\n123456789\n"


def test_compressed_record_reader_tracks_position_without_per_field_tell(tmp_path):
    """Avoid one buffered-file position query for every compressed field."""
    first_record = tuple(listtobytes(value) for value in (1, 2, 3, 4))
    second_record = tuple(listtobytes(value) for value in (5, 6, 7, 8))
    source = tmp_path / "compressed-records.bin"
    prefix = b"ignored-prefix"
    source.write_bytes(prefix + b"".join((*first_record, *second_record)))

    class CountingHandle:
        def __init__(self, handle):
            self.handle = handle
            self.tell_calls = 0

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            return self.handle.read(size)

        def seek(self, offset, whence=os.SEEK_SET):
            return self.handle.seek(offset, whence)

        def tell(self):
            self.tell_calls += 1
            return self.handle.tell()

    with source.open("rb") as raw_handle:
        raw_handle.seek(len(prefix))
        handle = CountingHandle(raw_handle)
        records = list(utils_module._iter_compressed_record_fields(handle, (2, 0)))

    assert records == [
        (first_record[2], first_record[0]),
        (second_record[2], second_record[0]),
    ]
    assert handle.tell_calls == 1


def test_compressed_record_reader_preserves_truncation_errors(tmp_path):
    """Retain header, payload, and mid-record corruption checks."""
    cases = (
        (b"short-header", "Truncated compressed block size"),
        ((10).to_bytes(64, byteorder="little") + b"short", "payload"),
        (listtobytes(1), "Incomplete compressed record"),
    )
    for index, (payload, message) in enumerate(cases):
        source = tmp_path / f"truncated-{index}.bin"
        source.write_bytes(payload)
        with source.open("rb") as handle:
            with pytest.raises(EOFError, match=message):
                list(utils_module._iter_compressed_record_fields(handle, (0,)))


def _write_manifest_fixture(output, *, reverse=False, changed_atom=False):
    """Write the same logical output with optionally reordered internal IDs."""
    molecule_rows = [
        ("A", [99 if changed_atom else 0, 1], [[0, 1, 1]], [(0, 2)]),
        ("B", [2], [], [(3, 3)]),
    ]
    if reverse:
        molecule_rows.reverse()
    store = _new_store(output)
    with store:
        for molecule_id, (species, atoms, bonds, ranges) in enumerate(
            molecule_rows,
            start=1,
        ):
            store.add_molecule(molecule_id, species, atoms, bonds, ranges)
        transitions = [
            (1, Counter({("A", "B"): 2, ("B", "C"): 1})),
            (0, Counter({("B", "C"): 3})),
        ]
        if reverse:
            transitions.reverse()
        for transition_index, events in transitions:
            store.stage_reaction_events(transition_index, events)
        store.finalize_reactions()
        store.finalize_and_publish()


def test_timed_output_manifest_is_independent_of_internal_id_order(tmp_path):
    """Compare logical content rather than nondeterministic molecule/type IDs."""
    first = tmp_path / "first.timeline.h5"
    second = tmp_path / "second.timeline.h5"
    _write_manifest_fixture(first)
    _write_manifest_fixture(second, reverse=True)

    first_manifest = build_timed_output_manifest(first, block_rows=2, block_bytes=64)
    second_manifest = build_timed_output_manifest(second, block_rows=2, block_bytes=64)
    differently_blocked = build_timed_output_manifest(
        first,
        block_rows=1,
        block_bytes=8,
    )

    assert (
        first_manifest["semantic_fingerprint"]
        == second_manifest["semantic_fingerprint"]
    )
    assert (
        first_manifest["semantic_fingerprint"]
        == differently_blocked["semantic_fingerprint"]
    )
    assert compare_timed_output_manifests(first_manifest, second_manifest) == []

    relocated_manifest = dict(second_manifest)
    relocated_manifest["fingerprints"] = {
        **second_manifest["fingerprints"],
        "sources": "relocated-source-paths",
    }
    assert compare_timed_output_manifests(first_manifest, relocated_manifest) == []
    assert compare_timed_output_manifests(
        first_manifest,
        relocated_manifest,
        include_provenance=True,
    )


def test_timed_output_manifest_canonicalizes_adjacent_ranges(tmp_path):
    """Treat split adjacent intervals as the same molecule-frame coverage."""
    merged = tmp_path / "merged.timeline.h5"
    split = tmp_path / "split.timeline.h5"
    for output, ranges in (
        (merged, [(0, 2)]),
        (split, [(0, 1), (2, 2)]),
    ):
        store = _new_store(output, reaction_enabled=False)
        with store:
            store.add_molecule(1, "A", [0], [], ranges)
            store.finalize_and_publish()

    merged_manifest = build_timed_output_manifest(merged)
    split_manifest = build_timed_output_manifest(split)

    assert merged_manifest["counts"]["molecule_range_count"] == 1
    assert split_manifest["counts"]["molecule_range_count"] == 2
    assert (
        merged_manifest["semantic_fingerprint"]
        == split_manifest["semantic_fingerprint"]
    )
    assert compare_timed_output_manifests(merged_manifest, split_manifest) == []


def test_timed_output_manifest_detects_semantic_change(tmp_path):
    """Change one atom ID and require the production comparison to fail."""
    baseline = tmp_path / "baseline.timeline.h5"
    changed = tmp_path / "changed.timeline.h5"
    _write_manifest_fixture(baseline)
    _write_manifest_fixture(changed, changed_atom=True)

    baseline_manifest = build_timed_output_manifest(baseline, block_rows=2)
    changed_manifest = build_timed_output_manifest(changed, block_rows=2)
    mismatches = compare_timed_output_manifests(changed_manifest, baseline_manifest)

    assert (
        changed_manifest["fingerprints"]["molecules"]
        != baseline_manifest["fingerprints"]["molecules"]
    )
    assert any("molecules" in mismatch for mismatch in mismatches)


def test_timed_output_manifest_rejects_corrupt_offsets_and_counts(tmp_path):
    """Fail closed when a complete file has inconsistent structural metadata."""
    output = tmp_path / "corrupt.timeline.h5"
    _write_manifest_fixture(output)
    with h5py.File(output, "r+") as handle:
        handle["molecules/atom_offsets"][-1] += 1

    with pytest.raises(ValueError, match="atom_offsets"):
        build_timed_output_manifest(output, block_rows=2)

    _write_manifest_fixture(output)
    with h5py.File(output, "r+") as handle:
        handle.attrs["reaction_event_row_count"] += 1

    with pytest.raises(ValueError, match="reaction_event_row_count"):
        build_timed_output_manifest(output, block_rows=2)


def test_timed_output_manifest_rejects_incomplete_and_corrupt_event_blocks(
    tmp_path,
):
    """Require a published status and an exact transition-to-event block index."""
    output = tmp_path / "invalid-events.timeline.h5"
    _write_manifest_fixture(output)
    with h5py.File(output, "r+") as handle:
        handle.attrs["status"] = "building"

    with pytest.raises(ValueError, match="not complete"):
        build_timed_output_manifest(output, block_rows=2)

    _write_manifest_fixture(output)
    with h5py.File(output, "r+") as handle:
        handle["reaction_events/block_start"][0] = 0

    with pytest.raises(ValueError, match="block disagrees"):
        build_timed_output_manifest(output, block_rows=2)


def test_timed_output_manifest_reads_large_tables_in_bounded_slices(
    tmp_path,
    monkeypatch,
):
    """Never materialize a complete molecule, range, frame, or event table."""
    output = tmp_path / "bounded.timeline.h5"
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(9)},
        frame_source={frame: (1, frame) for frame in range(9)},
    )
    with store:
        for molecule_id in range(1, 8):
            store.add_molecule(
                molecule_id,
                f"S{molecule_id}",
                [molecule_id],
                [],
                [(molecule_id - 1, molecule_id)],
            )
        for transition_index in range(8):
            store.stage_reaction_events(
                transition_index,
                Counter({(f"R{transition_index}", f"P{transition_index}"): 1}),
            )
        store.finalize_reactions()
        store.finalize_and_publish()

    reads = []
    original_read = timedoutputvalidate_module._read_dataset_slice

    def tracked_read(dataset, start, stop):
        reads.append((dataset.name, int(start), int(stop)))
        return original_read(dataset, start, stop)

    monkeypatch.setattr(
        timedoutputvalidate_module,
        "_read_dataset_slice",
        tracked_read,
    )
    manifest = build_timed_output_manifest(output, block_rows=2, block_bytes=64)

    assert manifest["counts"]["molecule_count"] == 7
    row_tables = {
        "/frames/source_id",
        "/molecules/molecule_id",
        "/molecule_ranges/molecule_id",
        "/reaction_events/reaction_id",
    }
    assert all(stop - start <= 2 for name, start, stop in reads if name in row_tables)
    assert all(
        start != 0 or stop != 7
        for name, start, stop in reads
        if name == "/molecules/molecule_id"
    )


def test_timed_output_check_cli_writes_and_compares_manifest(tmp_path, capsys):
    """Provide a cluster-friendly nonzero exit code for semantic mismatches."""
    baseline_hdf5 = tmp_path / "baseline.timeline.h5"
    reordered_hdf5 = tmp_path / "reordered.timeline.h5"
    changed_hdf5 = tmp_path / "changed.timeline.h5"
    baseline_json = tmp_path / "baseline.manifest.json"
    _write_manifest_fixture(baseline_hdf5)
    _write_manifest_fixture(reordered_hdf5, reverse=True)
    _write_manifest_fixture(changed_hdf5, changed_atom=True)

    assert (
        timed_output_check_main([str(baseline_hdf5), "--output", str(baseline_json)])
        == 0
    )
    assert (
        timed_output_check_main([str(reordered_hdf5), "--baseline", str(baseline_json)])
        == 0
    )
    assert (
        timed_output_check_main([str(changed_hdf5), "--baseline", str(baseline_json)])
        == 1
    )

    captured = capsys.readouterr()
    assert "fingerprints.molecules" in captured.err


def test_timed_output_check_cli_refuses_to_overwrite_input(tmp_path, capsys):
    """Protect the HDF5 artifact from an accidental --output collision."""
    output = tmp_path / "protected.timeline.h5"
    _write_manifest_fixture(output)

    assert timed_output_check_main([str(output), "--output", str(output)]) == 2
    assert build_timed_output_manifest(output)["status"] == "complete"
    assert "must not overwrite" in capsys.readouterr().err


def test_timed_output_check_cli_rejects_non_object_baseline(tmp_path, capsys):
    """Turn a syntactically valid but structurally invalid JSON baseline into exit 2."""
    output = tmp_path / "candidate.timeline.h5"
    baseline = tmp_path / "invalid-baseline.json"
    _write_manifest_fixture(output)
    baseline.write_text("[]\n", encoding="utf-8")

    assert (
        timed_output_check_main(
            [
                str(output),
                "--baseline",
                str(baseline),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    assert "must be JSON objects" in capsys.readouterr().err


def test_ordered_run_mp_keeps_workers_busy_past_slow_head(tmp_path):
    """Preserve result order without blocking task submission behind item zero."""
    records = list(
        run_mp(
            2,
            func=_head_of_line_task,
            l=range(12),
            unordered=False,
            chunksize=1,
            max_inflight=4,
            maxtasksperchild=None,
            disk_ordered=True,
            ordered_spool_dir=str(tmp_path),
            total=12,
            bar=False,
        )
    )

    assert [record[0] for record in records] == list(range(12))
    slow_head_finished = records[0][2]
    assert min(record[1] for record in records[4:]) < slow_head_finished
    assert not list(tmp_path.iterdir())


def test_ordered_run_mp_propagates_worker_error_and_cleans_spool(tmp_path):
    """Surface ordered worker errors without retaining temporary files."""
    with pytest.raises(RuntimeError, match="intentional ordered worker failure"):
        list(
            run_mp(
                2,
                func=_failing_timed_task,
                l=range(8),
                unordered=False,
                chunksize=1,
                max_inflight=4,
                maxtasksperchild=None,
                disk_ordered=True,
                ordered_spool_dir=str(tmp_path),
                total=8,
                bar=False,
            )
        )

    assert not list(tmp_path.iterdir())


def test_run_mp_worker_error_cancels_semaphore_producer():
    """Do not hang pool.join when a regular worker fails under backpressure."""
    with pytest.raises(RuntimeError, match="intentional ordered worker failure"):
        list(
            run_mp(
                2,
                func=_failing_timed_task,
                l=range(8),
                chunksize=1,
                max_inflight=4,
                maxtasksperchild=None,
                total=8,
                bar=False,
            )
        )


def test_run_mp_single_process_executes_inline(monkeypatch):
    """Avoid Pool startup and pipe serialization when only one CPU is requested."""

    def fail_if_pool_is_created(*args, **kwargs):
        raise AssertionError("nproc=1 must not create a multiprocessing Pool")

    monkeypatch.setattr(utils_module, "Pool", fail_if_pool_is_created)

    records = list(
        run_mp(
            1,
            func=lambda value: value + 1,
            l=range(3),
            chunksize=1,
            total=3,
            bar=False,
        )
    )

    assert records == [1, 2, 3]
    assert list(
        run_mp(
            1,
            func=lambda value: value,
            l=range(2),
            extra="context",
            chunksize=1,
            total=2,
            bar=False,
        )
    ) == [(0, "context"), (1, "context")]


def test_run_mp_does_not_start_more_workers_than_declared_tasks(monkeypatch):
    """Avoid idle worker startup when a stage exposes only a few work items."""
    worker_counts = []

    class InlinePool:
        def __init__(self, worker_count, **kwargs):
            worker_counts.append(worker_count)

        @staticmethod
        def imap_unordered(func, iterable, chunksize):
            return map(func, iterable)

        imap = imap_unordered

        @staticmethod
        def close():
            return None

        @staticmethod
        def terminate():
            return None

        @staticmethod
        def join():
            return None

    monkeypatch.setattr(utils_module, "Pool", InlinePool)

    records = list(
        run_mp(
            8,
            func=lambda value: value,
            l=range(2),
            initializer=lambda: None,
            chunksize=1,
            total=2,
            bar=False,
        )
    )

    assert records == [0, 1]
    assert worker_counts == [2]


def test_ordered_run_mp_early_close_stops_pool_and_cleans_spool(tmp_path):
    """Cancel outstanding workers when an ordered consumer stops early."""
    results = run_mp(
        2,
        func=_slow_tail_task,
        l=range(8),
        unordered=False,
        chunksize=1,
        max_inflight=4,
        maxtasksperchild=None,
        disk_ordered=True,
        ordered_spool_dir=str(tmp_path),
        total=8,
        bar=False,
    )

    assert next(results) == 0
    results.close()

    assert not list(tmp_path.iterdir())


def test_disk_ordered_spool_truncates_when_pending_results_are_drained(tmp_path):
    """Reclaim temporary disk when an ordered consumer catches up."""
    payload = np.arange(10_000, dtype=np.uint64)
    with _DiskOrderedResultSpool(2, str(tmp_path)) as spool:
        spool.put(1, payload)
        assert os.path.getsize(spool.data_path) == spool.write_offset
        assert spool.max_file_bytes > 0

        restored = spool.pop(1)

        np.testing.assert_array_equal(restored, payload)
        assert spool.pending_count == 0
        assert spool.write_offset == 0
        assert os.path.getsize(spool.data_path) == 0

    assert not list(tmp_path.iterdir())


def test_disk_ordered_spool_keeps_small_backlog_in_bounded_memory(tmp_path):
    """Avoid disk I/O until encoded reorder data exceeds a fixed byte budget."""
    with _DiskOrderedResultSpool(
        4,
        str(tmp_path),
        memory_limit_bytes=1024,
    ) as spool:
        spool.put(3, "late-small-result")

        assert spool.has(3)
        assert 0 < spool.memory_bytes <= 1024
        assert spool.max_memory_bytes == spool.memory_bytes
        assert spool.write_offset == 0
        assert not os.path.exists(spool.data_path)
        assert not os.path.exists(spool.index_path)

        assert spool.pop(3) == "late-small-result"
        assert spool.memory_bytes == 0
        assert spool.pending_count == 0


def test_disk_ordered_spool_spills_results_over_memory_budget(tmp_path):
    """Keep the in-memory reorder cache bounded even for one large result."""
    with _DiskOrderedResultSpool(
        2,
        str(tmp_path),
        memory_limit_bytes=256,
    ) as spool:
        spool.put(1, b"x" * 200)

        assert spool.memory_bytes == 0
        assert spool.max_memory_bytes == 0
        assert spool.write_offset > 0
        assert spool.bytes_written == spool.write_offset
        spool.data.flush()
        assert os.path.getsize(spool.data_path) == spool.write_offset
        assert spool.pop(1) == b"x" * 200


def test_disk_ordered_spool_restores_mixed_memory_and_disk_results(tmp_path):
    """Keep lookup and cleanup correct when one backlog spans both tiers."""
    with _DiskOrderedResultSpool(
        3,
        str(tmp_path),
        memory_limit_bytes=256,
    ) as spool:
        spool.put(1, "small")
        spool.put(2, b"x" * 200)

        assert spool.has(1)
        assert spool.has(2)
        assert spool.memory_bytes > 0
        assert spool.bytes_written > 0
        assert spool.pop(1) == "small"
        assert spool.pop(2) == b"x" * 200
        assert spool.pending_count == 0
        assert spool.memory_bytes == 0
        assert spool.write_offset == 0


def test_disk_ordered_spool_does_not_eagerly_zero_index(tmp_path, monkeypatch):
    """Do not dirty every mmap page merely to establish zero sentinels."""
    original_memmap = np.memmap
    full_slice_writes = []

    class TrackingMemmap:
        def __init__(self, *args, **kwargs):
            self.array = original_memmap(*args, **kwargs)

        def __getitem__(self, key):
            return self.array[key]

        def __setitem__(self, key, value):
            if key == slice(None):
                full_slice_writes.append(value)
            self.array[key] = value

        def flush(self):
            self.array.flush()

        @property
        def _mmap(self):
            return self.array._mmap

    monkeypatch.setattr(utils_module.np, "memmap", TrackingMemmap)
    with _DiskOrderedResultSpool(1_000_000, str(tmp_path)) as spool:
        assert not full_slice_writes
        assert not spool.has(999_999)

        spool.put(999_999, "late")
        assert spool.has(999_999)
        assert spool.pop(999_999) == "late"


def test_disk_ordered_spool_uses_raw_encoding_for_compact_results(
    tmp_path,
    monkeypatch,
):
    """Do not run pickle and LZ4 for compact SMILES-like result types."""
    values = ("[H][C]([H])[O]", b"route-bytes", None, "surrogate-\ud800")

    def fail_generic_codec(*args, **kwargs):
        raise AssertionError("compact strings must bypass the generic codec")

    monkeypatch.setattr(utils_module, "listtobytes", fail_generic_codec)
    monkeypatch.setattr(utils_module, "bytestolist", fail_generic_codec)
    with _DiskOrderedResultSpool(len(values), str(tmp_path)) as spool:
        for index, value in enumerate(values):
            spool.put(index, value)

        assert tuple(spool.pop(index) for index in range(len(values))) == values


def test_disk_ordered_spool_compresses_large_string_results(tmp_path):
    """Keep an upper bound on raw per-result spool growth."""
    value = "repeated-SMILES-" * 10_000
    with _DiskOrderedResultSpool(1, str(tmp_path)) as spool:
        spool.put(0, value)

        assert spool.bytes_written < len(value.encode("utf-8"))
        assert spool.pop(0) == value


def test_ordered_run_mp_rejects_inexact_result_count_and_cleans_spool(tmp_path):
    """Fail closed when declared ordering metadata exceeds real inputs."""
    with pytest.raises(RuntimeError, match="count does not match"):
        list(
            run_mp(
                1,
                func=int,
                l=range(3),
                unordered=False,
                chunksize=1,
                max_inflight=2,
                maxtasksperchild=None,
                disk_ordered=True,
                ordered_spool_dir=str(tmp_path),
                total=4,
                bar=False,
            )
        )

    assert not list(tmp_path.iterdir())


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


def test_molecule_output_reports_bounded_write_batches(tmp_path, monkeypatch):
    """Persist cross-boundary ranges with bounded real HDF5 appends."""
    output = tmp_path / "batched-molecules.timeline.h5"
    range_count = 4097
    append_counts = Counter()
    original_append = TimedOutputStore._append

    def tracked_append(dataset, values):
        if np.asarray(values).size:
            append_counts[dataset.name] += 1
        original_append(dataset, values)

    monkeypatch.setattr(TimedOutputStore, "_append", staticmethod(tracked_append))
    monkeypatch.setattr(
        timedoutput_module,
        "_MOLECULE_RANGE_BATCH_ROWS",
        4096,
    )
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(range_count)},
        frame_source={frame: (1, frame) for frame in range(range_count)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    starts = np.arange(range_count, dtype=np.uint64)
    with store:
        store.add_molecule(
            1,
            "A",
            [1],
            [],
            [(starts[:4096], starts[:4096]), (starts[4096:], starts[4096:])],
        )
        store.finalize_and_publish()

    metadata = read_timed_output_metadata(output)
    assert int(metadata["molecule_count"]) == 1
    assert int(metadata["molecule_range_count"]) == range_count
    assert int(metadata["molecule_write_batches"]) == 2
    assert int(metadata["maximum_molecule_batch_range_count"]) == 4096
    assert int(metadata["maximum_molecule_batch_definition_count"]) == 1
    assert int(metadata["molecule_range_batch_row_limit"]) == 4096
    assert int(metadata["timed_output_write_batch_byte_limit"]) == 64 * 1024**2
    assert float(metadata["timed_output_molecule_write_seconds"]) >= 0
    assert float(metadata["timed_output_reaction_write_seconds"]) == 0
    assert float(metadata["timed_output_write_seconds"]) == pytest.approx(
        float(metadata["timed_output_molecule_write_seconds"])
    )
    assert append_counts["/molecule_ranges/molecule_id"] == 2
    assert append_counts["/molecule_ranges/start_frame"] == 2
    assert append_counts["/molecule_ranges/end_frame"] == 2
    rows = list(iter_molecule_timeline(output))
    assert len(rows) == range_count
    assert rows[0] == (0, "A", "1", "")
    assert rows[-1] == (range_count - 1, "A", "1", "")


def test_range_heavy_molecule_output_coalesces_hdf5_appends(tmp_path, monkeypatch):
    """Avoid serial resize/write calls every 4,096 range rows."""
    output = tmp_path / "coalesced-ranges.timeline.h5"
    range_count = 16_385
    append_counts = Counter()
    original_append = TimedOutputStore._append

    def tracked_append(dataset, values):
        if np.asarray(values).size:
            append_counts[dataset.name] += 1
        original_append(dataset, values)

    monkeypatch.setattr(TimedOutputStore, "_append", staticmethod(tracked_append))
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(range_count)},
        frame_source={frame: (1, frame) for frame in range(range_count)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    starts = np.arange(range_count, dtype=np.uint64)
    with store:
        store.add_molecule(1, "A", [1], [], [(starts, starts)])
        store.finalize_and_publish()

    assert store.molecule_range_count == range_count
    assert store.molecule_write_batches == 1
    assert append_counts["/molecule_ranges/molecule_id"] == 1
    assert append_counts["/molecule_ranges/start_frame"] == 1
    assert append_counts["/molecule_ranges/end_frame"] == 1


def test_uint64_molecule_ranges_are_not_copied_before_batching(tmp_path):
    """Reuse production uint64 range blocks instead of casting them twice."""
    output = tmp_path / "zero-copy-ranges.timeline.h5"
    starts = np.array([0, 2, 4], dtype=np.uint64)
    ends = np.array([0, 2, 4], dtype=np.uint64)
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(5)},
        frame_source={frame: (1, frame) for frame in range(5)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    with store:
        store.add_molecule(1, "A", [1], [], [(starts, ends)])

        assert np.shares_memory(store._molecule_batch.range_starts[0], starts)
        assert np.shares_memory(store._molecule_batch.range_ends[0], ends)

        store.finalize_and_publish()


def test_molecule_range_ids_are_expanded_once_per_batch(tmp_path, monkeypatch):
    """Do not allocate one molecule-ID array for every small range block."""
    output = tmp_path / "compact-range-ids.timeline.h5"
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(5)},
        frame_source={frame: (1, frame) for frame in range(5)},
        molecule_enabled=True,
        reaction_enabled=False,
    )

    def fail_per_molecule_full(*_args, **_kwargs):
        raise AssertionError("range IDs must be expanded once when the batch flushes")

    with store:
        with monkeypatch.context() as patch:
            patch.setattr(timedoutput_module.np, "full", fail_per_molecule_full)
            store.add_molecule(
                7,
                "A",
                [1],
                [],
                [(np.array([0, 2, 4]), np.array([0, 2, 4]))],
            )
            store.add_molecule(
                9,
                "B",
                [2],
                [],
                [(np.array([1, 3]), np.array([1, 3]))],
            )
        assert store._molecule_batch.range_molecule_ids == [7, 9]
        assert store._molecule_batch.range_lengths == [3, 2]
        store.finalize_and_publish()

    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(
            handle["molecule_ranges/molecule_id"][:],
            np.array([7, 7, 7, 9, 9], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            handle["molecule_ranges/start_frame"][:],
            np.array([0, 2, 4, 1, 3], dtype=np.uint64),
        )


def test_bondless_molecules_do_not_stage_empty_bond_arrays(tmp_path):
    """Keep empty per-molecule bond arrays out of the bounded batch."""
    output = tmp_path / "bondless.timeline.h5"
    store = _new_store(
        output,
        timestep={0: 0},
        frame_source={0: (1, 0)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    with store:
        store.add_molecule(1, "[He]", [1], [], [(0, 0)])

        assert store._molecule_batch.bond_atom_arrays == []
        assert store._molecule_batch.bond_order_arrays == []
        assert store._molecule_batch.bond_offsets == [0]

        store.finalize_and_publish()

    with h5py.File(output, "r") as handle:
        assert handle["molecules/bond_atoms"].shape == (0, 2)
        assert handle["molecules/bond_order"].shape == (0,)
        np.testing.assert_array_equal(
            handle["molecules/bond_offsets"][:],
            np.array([0, 0], dtype=np.uint64),
        )


def test_uint64_molecule_ranges_skip_dtype_hierarchy_checks(tmp_path, monkeypatch):
    """Use the dtype kind fast path for normalized internal range arrays."""
    output = tmp_path / "dtype-kind-ranges.timeline.h5"
    starts = np.array([0, 2, 4], dtype=np.uint64)
    ends = np.array([0, 2, 4], dtype=np.uint64)
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(5)},
        frame_source={frame: (1, frame) for frame in range(5)},
        molecule_enabled=True,
        reaction_enabled=False,
    )

    def fail_dtype_hierarchy_check(*_args, **_kwargs):
        raise AssertionError("normalized ranges must use dtype.kind")

    with store:
        with monkeypatch.context() as patch:
            patch.setattr(
                timedoutput_module.np,
                "issubdtype",
                fail_dtype_hierarchy_check,
            )
            store.add_molecule(1, "A", [1], [], [(starts, ends)])
        store.finalize_and_publish()

    assert list(iter_molecule_timeline(output)) == [
        (0, "A", "1", ""),
        (2, "A", "1", ""),
        (4, "A", "1", ""),
    ]


def test_single_molecule_range_skips_numpy_reductions(tmp_path, monkeypatch):
    """Validate the dominant one-range case without four NumPy dispatches."""
    output = tmp_path / "single-range.timeline.h5"
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(5)},
        frame_source={frame: (1, frame) for frame in range(5)},
        molecule_enabled=True,
        reaction_enabled=False,
    )

    def fail_reduction(*_args, **_kwargs):
        raise AssertionError("one range must use scalar validation")

    with store:
        with monkeypatch.context() as patch:
            patch.setattr(timedoutput_module.np, "any", fail_reduction)
            store.add_molecule(
                1,
                "A",
                [1],
                [],
                [(np.array([2], dtype=np.uint64), np.array([4], dtype=np.uint64))],
            )
        store.finalize_and_publish()

    assert list(iter_molecule_timeline(output)) == [
        (2, "A", "1", ""),
        (3, "A", "1", ""),
        (4, "A", "1", ""),
    ]


@pytest.mark.parametrize(
    ("starts", "ends"),
    [
        (np.array([-1], dtype=np.int64), np.array([0], dtype=np.int64)),
        (np.array([0], dtype=np.int64), np.array([-1], dtype=np.int64)),
        (np.array([2], dtype=np.uint64), np.array([1], dtype=np.uint64)),
        (np.array([0], dtype=np.uint64), np.array([5], dtype=np.uint64)),
    ],
)
def test_molecule_range_validation_survives_zero_copy_path(
    tmp_path,
    starts,
    ends,
):
    """Reject signed and unsigned invalid ranges before publishing output."""
    output = tmp_path / "invalid-ranges.timeline.h5"
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(5)},
        frame_source={frame: (1, frame) for frame in range(5)},
        molecule_enabled=True,
        reaction_enabled=False,
    )

    with pytest.raises(RuntimeError, match="invalid molecule range"), store:
        store.add_molecule(1, "A", [1], [], [(starts, ends)])

    assert not output.exists()


def test_molecule_byte_limit_flushes_before_next_range(tmp_path, monkeypatch):
    """Do not let the next divisible range row overrun the byte limit."""
    output = tmp_path / "byte-bounded-molecules.timeline.h5"
    byte_limit = 80
    flushed_byte_counts = []
    original_flush = TimedOutputStore._flush_molecule_batch

    def tracked_flush(store):
        if store._molecule_batch.has_data():
            flushed_byte_counts.append(store._molecule_batch.byte_count)
        original_flush(store)

    monkeypatch.setattr(timedoutput_module, "_WRITE_BATCH_BYTES", byte_limit)
    monkeypatch.setattr(TimedOutputStore, "_flush_molecule_batch", tracked_flush)
    store = _new_store(
        output,
        timestep={0: 0, 1: 1},
        frame_source={0: (1, 0), 1: (1, 1)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    with store:
        store.add_molecule(
            1,
            "A",
            [1],
            [],
            [(np.array([0, 1]), np.array([0, 1]))],
        )
        store.finalize_and_publish()

    assert flushed_byte_counts == [65, 24]
    assert all(byte_count <= byte_limit for byte_count in flushed_byte_counts)


def test_molecule_byte_limit_flushes_before_next_definition(tmp_path, monkeypatch):
    """Do not combine individually bounded molecules into an oversized batch."""
    output = tmp_path / "byte-bounded-definitions.timeline.h5"
    byte_limit = 80
    flushed_byte_counts = []
    original_flush = TimedOutputStore._flush_molecule_batch

    def tracked_flush(store):
        if store._molecule_batch.has_data():
            flushed_byte_counts.append(store._molecule_batch.byte_count)
        original_flush(store)

    monkeypatch.setattr(timedoutput_module, "_WRITE_BATCH_BYTES", byte_limit)
    monkeypatch.setattr(TimedOutputStore, "_flush_molecule_batch", tracked_flush)
    store = _new_store(
        output,
        timestep={0: 0},
        frame_source={0: (1, 0)},
        molecule_enabled=True,
        reaction_enabled=False,
    )
    with store:
        store.add_molecule(1, "A", [1], [], [])
        store.add_molecule(2, "A", [2], [], [])
        store.finalize_and_publish()

    assert flushed_byte_counts == [41, 40]
    assert all(byte_count <= byte_limit for byte_count in flushed_byte_counts)


def test_reaction_output_reports_bounded_write_batches(tmp_path, monkeypatch):
    """Persist cross-boundary reactions with bounded real HDF5 appends."""
    output = tmp_path / "batched-reactions.timeline.h5"
    transition_count = 4097
    append_counts = Counter()
    original_append = TimedOutputStore._append

    def tracked_append(dataset, values):
        if np.asarray(values).size:
            append_counts[dataset.name] += 1
        original_append(dataset, values)

    monkeypatch.setattr(TimedOutputStore, "_append", staticmethod(tracked_append))
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(transition_count + 1)},
        frame_source={frame: (1, frame) for frame in range(transition_count + 1)},
        molecule_enabled=False,
        reaction_enabled=True,
    )
    with store:
        for transition_index in range(transition_count):
            store.stage_reaction_events(
                transition_index,
                Counter({("A", "B"): 1}),
            )
        store.finalize_reactions()
        store.finalize_and_publish()

    metadata = read_timed_output_metadata(output)
    assert int(metadata["reaction_type_count"]) == 1
    assert int(metadata["reaction_event_row_count"]) == transition_count
    assert int(metadata["reaction_write_batches"]) == 2
    assert append_counts["/reaction_events/transition_index"] == 2
    assert append_counts["/reaction_events/reaction_id"] == 2
    assert append_counts["/reaction_events/count"] == 2
    rows = list(iter_reaction_events(output))
    assert len(rows) == transition_count
    assert rows[0] == (0, "A", "B")
    assert rows[-1] == (transition_count - 1, "A", "B")


def test_reaction_type_byte_limit_flushes_before_next_type(tmp_path, monkeypatch):
    """Do not append a reaction type after it would overrun the byte limit."""
    output = tmp_path / "byte-bounded-reaction-types.timeline.h5"
    byte_limit = 100
    flushed_byte_counts = []
    original_flush = TimedOutputStore._flush_reaction_type_batch

    def tracked_flush(store):
        if store._reaction_batch.reactants:
            flushed_byte_counts.append(store._reaction_batch.type_bytes)
        original_flush(store)

    monkeypatch.setattr(timedoutput_module, "_WRITE_BATCH_BYTES", byte_limit)
    monkeypatch.setattr(
        TimedOutputStore,
        "_flush_reaction_type_batch",
        tracked_flush,
    )
    store = _new_store(
        output,
        timestep={0: 0, 1: 1},
        frame_source={0: (1, 0), 1: (1, 1)},
        molecule_enabled=False,
        reaction_enabled=True,
    )
    events = Counter(
        {
            ("A" * 26, "B" * 26): 1,
            ("C" * 26, "D" * 26): 1,
        }
    )
    with store:
        store.stage_reaction_events(0, events)
        store.finalize_reactions()
        store.finalize_and_publish()

    assert flushed_byte_counts == [60, 60]
    assert all(byte_count <= byte_limit for byte_count in flushed_byte_counts)
    assert list(iter_reaction_events(output)) == [
        (0, "A" * 26, "B" * 26),
        (0, "C" * 26, "D" * 26),
    ]


def test_reverse_transition_blocks_survive_multiple_write_batches(tmp_path):
    """Restore transition order after reverse submission crosses a write batch."""
    output = tmp_path / "reverse-batched-reactions.timeline.h5"
    transition_count = 5000
    store = _new_store(
        output,
        timestep={frame: frame for frame in range(transition_count + 1)},
        frame_source={frame: (1, frame) for frame in range(transition_count + 1)},
        molecule_enabled=False,
        reaction_enabled=True,
    )
    with store:
        for transition_index in reversed(range(transition_count)):
            store.stage_reaction_events(
                transition_index,
                Counter({("A", "B"): 1}),
            )
        store.finalize_reactions()
        store.finalize_and_publish()

    with h5py.File(output, "r") as handle:
        events = handle["reaction_events"]
        np.testing.assert_array_equal(
            events["block_start"][:],
            np.arange(transition_count - 1, -1, -1, dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            events["block_length"][:],
            np.ones(transition_count, dtype=np.uint64),
        )
    rows = list(iter_reaction_events(output))
    assert len(rows) == transition_count
    assert rows[0] == (0, "A", "B")
    assert rows[-1] == (transition_count - 1, "A", "B")


def test_single_transition_reactions_are_split_into_bounded_batches(
    tmp_path,
    monkeypatch,
):
    """Split one large transition without breaking its block index."""
    output = tmp_path / "large-transition.timeline.h5"
    reaction_count = 4097
    append_counts = Counter()
    original_append = TimedOutputStore._append

    def tracked_append(dataset, values):
        if np.asarray(values).size:
            append_counts[dataset.name] += 1
        original_append(dataset, values)

    monkeypatch.setattr(TimedOutputStore, "_append", staticmethod(tracked_append))
    store = _new_store(
        output,
        timestep={0: 0, 1: 1},
        frame_source={0: (1, 0), 1: (1, 1)},
        molecule_enabled=False,
        reaction_enabled=True,
    )
    events = Counter(
        {
            (f"reactant-{index}", f"product-{index}"): 1
            for index in range(reaction_count)
        }
    )
    with store:
        store.stage_reaction_events(0, events)
        store.finalize_reactions()
        store.finalize_and_publish()

    metadata = read_timed_output_metadata(output)
    assert int(metadata["reaction_event_row_count"]) == reaction_count
    assert int(metadata["reaction_write_batches"]) == 2
    assert append_counts["/reaction_events/transition_index"] == 2
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(
            handle["reaction_events/block_start"][:],
            [0],
        )
        np.testing.assert_array_equal(
            handle["reaction_events/block_length"][:],
            [reaction_count],
        )


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

    output = tmp_path / "multi-source.timeline.h5"
    store = TimedOutputStore(
        str(output),
        cache_mib=1,
        input_filenames=[str(filename) for filename in files],
        timestep=rng.timestep,
        frame_source=rng.framesource,
        stepinterval=2,
        molecule_enabled=False,
        reaction_enabled=False,
    )
    with store:
        store.finalize_and_publish()
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["frames/source_id"][:], [1, 3])
        np.testing.assert_array_equal(handle["frames/source_frame"][:], [0, 0])
        np.testing.assert_array_equal(
            handle["frames/timestep"][:],
            list(rng.timestep),
        )


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


def test_molecule_name_table_deduplicates_long_species_names():
    """Store one long species name once instead of padding every molecule row."""
    long_name = "C" * 1000
    expanded = [long_name, "[H]", long_name, "[H]"] * 100

    table = path_module._MoleculeNameTable.from_names(expanded)

    assert len(table) == len(expanded)
    assert len(table.names) == 2
    assert table.ids.dtype == np.dtype(np.uint8)
    assert list(map(str, table)) == expanded
    np.testing.assert_array_equal(table[[0, 1, 2]], expanded[:3])
    assert table.ids.nbytes + table.names.nbytes < len(expanded) * len(long_name)


def test_reaction_matrix_counts_compact_name_ids_without_expanding_names():
    """Aggregate routes on compact IDs and decode only the unique pairs."""
    table = path_module._MoleculeNameTable.from_names(["A", "B", "A", "C"])
    generator = object.__new__(_GenerateMatrix)
    generator.mname = table
    routes = np.array([[1, 2], [3, 2], [2, 4], [1, 1]])

    counts = {tuple(pair): count for pair, count in generator._getallroute(routes)}
    retained_counts = {
        tuple(pair): count
        for pair, count in generator._getallroute(Counter({(0, 1): 2, (1, 2): 1}))
    }

    assert counts == {("A", "B"): 2, ("B", "C"): 1}
    assert retained_counts == counts


def test_species_output_does_not_allocate_one_counter_per_frame(
    tmp_path,
    monkeypatch,
):
    """Keep the default species timeline off the per-frame Python-object path."""
    frame_count = 4097
    frame_block = listtobytes(np.arange(frame_count, dtype=np.uint64))
    empty_block = listtobytes([])
    molecule_record = b"".join(
        (listtobytes([0]), empty_block, empty_block, frame_block)
    )
    molecule_file = tmp_path / "molecules.bin"
    molecule_file.write_bytes(molecule_record * 2)
    output = tmp_path / "species.out"
    generator = object.__new__(_GenerateMatrix)
    generator.moleculetemp2filename = str(molecule_file)
    generator.speciesfilename = str(output)
    generator.timestep = np.arange(frame_count, dtype=np.uint64)
    generator.mname = path_module._MoleculeNameTable.from_names(["A", "B"])
    counter_count = 0

    class TrackedCounter(Counter):
        def __init__(self, *args, **kwargs):
            nonlocal counter_count
            counter_count += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(matrix_module, "Counter", TrackedCounter)

    generator._printspecies()

    assert counter_count <= len(generator.mname.names)
    with output.open() as handle:
        first = handle.readline()
        for line in handle:
            last = line
    assert first == "Timestep 0: A 1 B 1\n"
    assert last == f"Timestep {frame_count - 1}: A 1 B 1\n"


@pytest.mark.parametrize(
    ("byte_budget", "spool_budget", "expected_mode", "expected_reused"),
    [
        (1024, 1024, "dense", False),
        (80, 1024, "sparse", True),
        (80, 1, "sparse", False),
    ],
)
def test_species_output_adapts_storage_and_matches_counter_semantics(
    tmp_path,
    monkeypatch,
    byte_budget,
    spool_budget,
    expected_mode,
    expected_reused,
):
    """Choose the smaller timeline layout without changing counts or ordering."""
    names = ["B", "A", "B", "C", "D", "E", "F", "G"]
    frames_by_molecule = [
        np.array([0, 0, 2, 4], dtype=np.uint8),
        np.array([0, 1], dtype=np.uint8),
        np.array([0, 1, 4], dtype=np.uint8),
        np.array([7], dtype=np.uint8),
        np.array([8], dtype=np.uint8),
        np.array([9], dtype=np.uint8),
        np.array([10], dtype=np.uint8),
        np.array([11], dtype=np.uint8),
    ]
    frame_count = 12
    empty_block = listtobytes([])
    records = []
    for frames in frames_by_molecule:
        records.extend(
            (listtobytes([0]), empty_block, empty_block, listtobytes(frames))
        )
    molecule_file = tmp_path / f"{expected_mode}-molecules.bin"
    molecule_file.write_bytes(b"".join(records))
    output = tmp_path / f"{expected_mode}-species.out"
    generator = object.__new__(_GenerateMatrix)
    generator.moleculetemp2filename = str(molecule_file)
    generator.speciesfilename = str(output)
    generator.timestep = np.arange(100, 100 + frame_count, dtype=np.uint64)
    generator.mname = path_module._MoleculeNameTable.from_names(names)
    reference = [Counter() for _ in range(frame_count)]
    for name, frames in zip(names, frames_by_molecule):
        for frame in frames:
            reference[int(frame)][name] += 1
    expected = "".join(
        f"Timestep {100 + frame}:"
        + "".join(f" {name} {count}" for name, count in row.items())
        + "\n"
        for frame, row in enumerate(reference)
    )
    observed_modes = []
    observed_reuse = []
    observed_block_bytes = []
    observed_block_frame_counts = []
    temporary_paths = []
    observation_paths = []
    original_store = matrix_module._SpeciesTimelineStore

    class TrackingStore(original_store):
        def allocate(self):
            super().allocate()
            observed_modes.append(self.mode)
            observed_reuse.append(self.observation_reused)
            observed_block_bytes.append(self.maximum_count_block_bytes)
            observed_block_frame_counts.append(self.block_frame_count)
            temporary_paths.append(self.path)
            observation_paths.append(self.observation_path)

    monkeypatch.setattr(matrix_module, "_SPECIES_COUNT_BLOCK_BYTES", byte_budget)
    monkeypatch.setattr(matrix_module, "_SPECIES_EVENT_CHUNK_ROWS", 2)
    monkeypatch.setattr(
        matrix_module,
        "_SPECIES_OBSERVATION_SPOOL_BYTES",
        spool_budget,
    )

    if expected_mode == "sparse":

        def fail_on_python_counter(*args, **kwargs):
            raise AssertionError(
                "Sparse species aggregation must not build a Python Counter"
            )

        monkeypatch.setattr(matrix_module, "Counter", fail_on_python_counter)

    monkeypatch.setattr(matrix_module, "_SpeciesTimelineStore", TrackingStore)

    generator._printspecies()

    assert observed_modes == [expected_mode]
    assert observed_reuse == [expected_reused]
    assert observed_block_bytes[0] <= byte_budget or observed_block_frame_counts == [1]
    assert observation_paths == [None]
    assert all(path is None or not os.path.exists(path) for path in temporary_paths)
    assert output.read_text() == expected


def test_sparse_species_spool_population_does_not_rescan_each_frame_block(
    monkeypatch,
):
    """Populate many sparse partitions with one vectorized stable grouping."""
    frame_count = 4096
    original_count_nonzero = matrix_module.np.count_nonzero
    count_nonzero_calls = 0

    def tracked_count_nonzero(*args, **kwargs):
        nonlocal count_nonzero_calls
        count_nonzero_calls += 1
        return original_count_nonzero(*args, **kwargs)

    monkeypatch.setattr(matrix_module, "_SPECIES_COUNT_BLOCK_BYTES", 4096)
    monkeypatch.setattr(matrix_module, "_SPECIES_EVENT_CHUNK_ROWS", frame_count)
    monkeypatch.setattr(matrix_module.np, "count_nonzero", tracked_count_nonzero)

    with matrix_module._SpeciesTimelineStore(
        frame_count,
        frame_count,
        frame_count,
    ) as store:
        for index in range(frame_count):
            store.observe(index, index, np.asarray([index], dtype=np.uint16))
        store.allocate()

        assert store.mode == "sparse"
        assert store.observation_reused is True
        assert count_nonzero_calls <= 1


def test_smiles_worker_returns_only_the_species_name(tmp_path):
    """Keep all molecule arrays out of the worker-to-parent IPC result."""
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        printmoleculetime=True,
    )
    collector = _CollectSMILESPaths(rng)
    collector.atomtype = np.array([0])
    collector.atomnames = np.array(["H"])
    line = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
    )

    result = collector._calmoleculeSMILESname(line)

    assert isinstance(result, str)
    path_module._initialize_smiles_worker(collector.atomname, np.array([0]))
    assert path_module._get_smiles_name(line) == result


def test_smiles_decoded_worker_returns_reusable_structure(tmp_path):
    """Let the verified cheap-fork path reuse the worker's decoded arrays."""
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
    )
    collector = _CollectSMILESPaths(rng)
    collector.atomtype = np.array([0])
    collector.atomnames = np.array(["H"])
    line = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
    )

    path_module._initialize_smiles_worker(collector.atomname, np.array([0]))
    name, atoms, bonds = path_module._get_smiles_name_and_structure(line)

    assert name == collector._calmoleculeSMILESname(line)
    np.testing.assert_array_equal(atoms, [0])
    assert atoms.dtype == np.uint64
    assert bonds == []


def test_smiles_parent_result_records_enforce_structure_states():
    """Represent compressed and decoded parent payloads without tuple sentinels."""
    compressed = (b"atoms", b"pairs", b"levels", b"frames")

    parent_result = path_module._SmilesResultRecord.from_compressed(
        "[H]",
        compressed,
    )
    decoded_result = path_module._SmilesResultRecord.from_decoded(
        "[H]",
        np.array([0], dtype=np.int64),
        [],
        compressed[3],
    )

    assert parent_result.structure_record == compressed
    assert parent_result.frame_block == compressed[3]
    assert parent_result.atoms is None
    assert parent_result.bonds is None
    assert decoded_result.structure_record is None
    np.testing.assert_array_equal(decoded_result.atoms, [0])
    assert decoded_result.bonds == []
    assert decoded_result.frame_block == compressed[3]
    with pytest.raises(ValueError, match="3 or 4 fields"):
        path_module._SmilesResultRecord.from_compressed("[H]", compressed[:2])
    with pytest.raises(ValueError, match="require atoms and bonds"):
        path_module._SmilesResultRecord.from_decoded("[H]", None, [])


@pytest.mark.parametrize(
    ("name", "atoms", "bonds", "expected"),
    [
        ("[H]", np.array([0], dtype=np.uint64), [], "[H] 0 "),
        (
            "C=C",
            np.array([10, 2], dtype=np.int64),
            [[10, 2, 2], [2, 7, 1]],
            "C=C 10;2 10,2,2;2,7,1",
        ),
        ("", np.array([], dtype=np.int64), [], "  "),
    ],
)
def test_molecule_name_format_avoids_generic_recursive_formatter(
    monkeypatch,
    name,
    atoms,
    bonds,
    expected,
):
    """Keep the hot `.moname` encoder on its specialized linear path."""
    collector = object.__new__(_CollectSMILESPaths)

    def fail_generic_formatter(*_args, **_kwargs):
        raise AssertionError("generic recursive formatter must not be used")

    monkeypatch.setattr(
        path_module,
        "listtostirng",
        fail_generic_formatter,
        raising=False,
    )

    assert collector._formatmoleculename(name, atoms, bonds) == expected


def test_smiles_radical_pattern_is_built_once_per_converter(monkeypatch):
    """Do not rebuild the atom-name regular expression for every molecule."""
    collector = object.__new__(_CollectSMILESPaths)
    collector.atomname = np.asarray(["C", "H", "Cl", "Na"])

    assert collector._re("C") == "[C]"

    def fail_repeated_sort(*_args, **_kwargs):
        raise AssertionError("SMILES atom pattern must be cached")

    monkeypatch.setattr(builtins, "sorted", fail_repeated_sort)
    assert collector._re("[H]c(Cl)C([H])Cl") == "[H][c]([Cl])[C]([H])[Cl]"


def test_smiles_converter_reuses_empty_rdkit_template(monkeypatch):
    """Parse the empty RDKit molecule once instead of once per structure."""
    collector = object.__new__(_CollectSMILESPaths)
    collector.atomname = np.asarray(["C"])
    collector.atomnames = np.asarray(["C"])
    original_mol_from_smiles = path_module.Chem.MolFromSmiles
    calls = 0

    def counted_mol_from_smiles(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_mol_from_smiles(*args, **kwargs)

    monkeypatch.setattr(
        path_module.Chem,
        "MolFromSmiles",
        counted_mol_from_smiles,
    )

    assert collector.convertSMILES(np.asarray([0]), []) == "[C]"
    assert collector.convertSMILES(np.asarray([0]), []) == "[C]"
    assert calls == 1


def test_smiles_name_cache_reuses_equivalent_local_structure(monkeypatch):
    """Reuse a species name when only the molecule's global atom IDs differ."""
    collector = object.__new__(_CollectSMILESPaths)
    collector.atomname = np.asarray(["C", "H"])
    collector.atomtype = np.asarray([0, 1, 0, 1])
    collector.atomnames = collector.atomname[collector.atomtype]
    calls = []

    def counted_convert(atoms, bonds):
        calls.append((np.asarray(atoms).copy(), list(bonds)))
        return "[H][C]"

    monkeypatch.setattr(collector, "convertSMILES", counted_convert)

    first = collector._calmoleculeSMILESname_from_decoded(
        np.asarray([0, 1]),
        [[0, 1, 1]],
    )
    second = collector._calmoleculeSMILESname_from_decoded(
        np.asarray([2, 3]),
        [[2, 3, 1]],
    )

    assert first == second == "[H][C]"
    assert len(calls) == 1


def test_smiles_name_cache_is_byte_bounded_and_keeps_bond_levels(monkeypatch):
    """Evict old names within a fixed budget and never merge unlike bonds."""
    collector = object.__new__(_CollectSMILESPaths)
    collector.atomname = np.asarray(["C", "H"])
    collector.atomtype = np.asarray([0, 1, 0, 1])
    collector.atomnames = collector.atomname[collector.atomtype]
    monkeypatch.setattr(path_module, "_SMILES_NAME_CACHE_BYTES", 600)
    calls = 0

    def counted_convert(_atoms, bonds):
        nonlocal calls
        calls += 1
        return f"bond-{bonds[0][2]}"

    monkeypatch.setattr(collector, "convertSMILES", counted_convert)

    first = collector._calmoleculeSMILESname_from_decoded(
        np.asarray([0, 1]),
        [[0, 1, 1]],
    )
    second = collector._calmoleculeSMILESname_from_decoded(
        np.asarray([2, 3]),
        [[2, 3, 2]],
    )
    repeated = collector._calmoleculeSMILESname_from_decoded(
        np.asarray([0, 1]),
        [[0, 1, 1]],
    )

    assert (first, second, repeated) == ("bond-1", "bond-2", "bond-1")
    assert calls == 3
    assert collector._smiles_name_cache_bytes <= 600


def test_oversized_smiles_cache_key_is_rejected_before_allocation(monkeypatch):
    """Do not duplicate one structure whose cache key cannot fit the budget."""
    collector = object.__new__(_CollectSMILESPaths)
    collector.atomname = np.asarray(["C"])
    collector.atomtype = np.asarray([0])
    collector.atomnames = collector.atomname[collector.atomtype]
    monkeypatch.setattr(path_module, "_SMILES_NAME_CACHE_BYTES", 512)
    calls = 0

    def counted_convert(_atoms, _bonds):
        nonlocal calls
        calls += 1
        return "[C]"

    def fail_key_array(*_args, **_kwargs):
        raise AssertionError("an oversized cache key must not allocate bond storage")

    monkeypatch.setattr(collector, "convertSMILES", counted_convert)
    monkeypatch.setattr(path_module.np, "empty", fail_key_array)

    assert collector._calmoleculeSMILESname_from_decoded(np.asarray([0]), []) == "[C]"
    assert collector._calmoleculeSMILESname_from_decoded(np.asarray([0]), []) == "[C]"
    assert calls == 2


def test_short_molecule_ranges_use_linear_scan_without_numpy_setup(monkeypatch):
    """Coalesce common short frame lists without per-molecule NumPy kernels."""
    collector = object.__new__(_CollectSMILESPaths)
    collector._moleculeframefilter = None
    collector._moleculetimestepfilter = None

    def fail_vector_scan(*_args, **_kwargs):
        raise AssertionError("short frame lists must not call numpy.diff")

    monkeypatch.setattr(path_module.np, "diff", fail_vector_scan)
    ranges = list(
        collector._itermoleculeranges(np.asarray([0, 1, 1, 3, 4, 7], dtype=np.uint16))
    )

    assert len(ranges) == 1
    starts, ends = ranges[0]
    np.testing.assert_array_equal(starts, [0, 3, 7])
    np.testing.assert_array_equal(ends, [1, 4, 7])
    assert starts.dtype == ends.dtype == np.uint64


def test_smiles_fork_pool_reuses_worker_decoding(tmp_path, monkeypatch):
    """Read only frame payloads after cheap fork workers return structures."""
    blocks = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
        listtobytes(np.array([0], dtype=np.uint64)),
    )
    molecule_temp = tmp_path / "fork-decoded-smiles.bin"
    molecule_temp.write_bytes(b"".join(blocks) * 2)
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=8,
        printmoleculetime=True,
    )
    rng.hmmit = 2
    structure_bytes = sum(len(block) for block in blocks[:3])
    rng.smilesworkmetrics = hmmfilter_module._SmilesWorkMetrics(
        total_compressed_bytes=2 * structure_bytes,
        max_compressed_record_bytes=structure_bytes,
    )
    rng.moleculetemp2filename = str(molecule_temp)
    rng.moleculefilename = str(tmp_path / "fork-decoded-smiles.txt")
    rng.atomtype = np.array([0])
    captured = {}
    selected_field_groups = []
    original_iter_molecule_fields = path_module._iter_molecule_fields

    def tracked_iter_molecule_fields(handle, selected_fields, **kwargs):
        selected_field_groups.append(tuple(selected_fields))
        return original_iter_molecule_fields(handle, selected_fields, **kwargs)

    def fake_run_mp(nproc, **kwargs):
        captured["nproc"] = nproc
        captured.update(kwargs)
        return iter([("[H]", np.array([0], dtype=np.int64), []) for _ in range(2)])

    monkeypatch.setattr(path_module, "get_start_method", lambda: "fork")
    monkeypatch.setattr(path_module, "_smiles_worker_count", lambda *args: 2)
    monkeypatch.setattr(path_module, "_smiles_pool_limits", lambda *args: (64, 256))
    monkeypatch.setattr(path_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(
        path_module,
        "_iter_molecule_fields",
        tracked_iter_molecule_fields,
    )
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])

    def fail_parent_structure_decode(_record):
        raise AssertionError("parent must reuse decoded cheap-fork structures")

    monkeypatch.setattr(collector, "_getatomsandbonds", fail_parent_structure_decode)

    collector._printmoleculename(None)

    assert captured["nproc"] == 2
    assert captured["func"] is path_module._get_smiles_name_and_structure
    assert selected_field_groups == [(0, 1, 2), (3,)]
    assert Path(rng.moleculefilename).read_text().count("\n") == 2


def test_smiles_stage_does_not_recycle_heavy_workers(tmp_path, monkeypatch):
    """Do not repeatedly reload RDKit/OpenBabel during one molecule stage."""
    blocks = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
        listtobytes(np.array([0], dtype=np.uint64)),
    )
    molecule_temp = tmp_path / "molecules.bin"
    molecule_temp.write_bytes(b"".join(blocks) * 2)
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=2,
    )
    rng.hmmit = 2
    rng.moleculetemp2filename = str(molecule_temp)
    rng.moleculefilename = str(tmp_path / "molecules.txt")
    rng.atomtype = np.array([0])
    captured = {}

    def fake_run_mp(nproc, **kwargs):
        captured.update(kwargs)
        return iter(["[H]", "[H]"])

    monkeypatch.setattr(path_module, "run_mp", fake_run_mp)
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])

    collector._printmoleculename(None)

    assert captured["maxtasksperchild"] is None
    assert captured["func"] is path_module._get_smiles_name
    assert captured["initializer"] is path_module._initialize_smiles_worker
    np.testing.assert_array_equal(captured["initargs"][0], collector.atomname)
    np.testing.assert_array_equal(captured["initargs"][1], collector.atomtype)


@pytest.mark.parametrize(
    ("maximum_structure_bytes", "reuse_worker_structures"),
    [
        (64 * 1024, True),
        (64 * 1024 + 1, False),
    ],
)
def test_smiles_fork_pool_batches_cheap_records(
    tmp_path,
    monkeypatch,
    maximum_structure_bytes,
    reuse_worker_structures,
):
    """Batch by average size without returning an oversized decoded outlier."""
    structure = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
    )
    record = b"".join((*structure, listtobytes(np.array([0], dtype=np.uint64))))
    molecule_temp = tmp_path / "fork-smiles.bin"
    molecule_count = 128
    molecule_temp.write_bytes(record * molecule_count)
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=8,
    )
    rng.hmmit = molecule_count
    rng.smilesworkmetrics = hmmfilter_module._SmilesWorkMetrics(
        total_compressed_bytes=molecule_count * sum(len(block) for block in structure),
        max_compressed_record_bytes=maximum_structure_bytes,
    )
    rng.moleculetemp2filename = str(molecule_temp)
    rng.moleculefilename = str(tmp_path / "fork-smiles.txt")
    rng.atomtype = np.array([0])
    captured = {}

    def fake_run_mp(nproc, **kwargs):
        captured["nproc"] = nproc
        captured.update(kwargs)
        if reuse_worker_structures:
            assert kwargs["func"] is path_module._get_smiles_name_and_structure
            return iter(
                [
                    ("[H]", np.array([0], dtype=np.int64), [])
                    for _ in range(molecule_count)
                ]
            )
        assert kwargs["func"] is path_module._get_smiles_name
        return iter(["[H]"] * molecule_count)

    monkeypatch.setattr(path_module, "get_start_method", lambda: "fork")
    monkeypatch.setattr(path_module, "_smiles_worker_count", lambda *args: 2)
    monkeypatch.setattr(path_module, "run_mp", fake_run_mp)
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])

    collector._printmoleculename(None)

    assert captured["nproc"] == 2
    assert captured["func"] is (
        path_module._get_smiles_name_and_structure
        if reuse_worker_structures
        else path_module._get_smiles_name
    )
    assert captured["chunksize"] == 64
    assert captured["max_inflight"] == 256


@pytest.mark.parametrize(
    ("workers", "molecules", "structure_bytes", "start_method", "expected"),
    [
        (2, 112_300, 38_100_340, "fork", (64, 256)),
        (4, 280_750, 95_250_850, "fork", (64, 512)),
        (2, 112_300, 38_100_340, "spawn", (1, 4)),
        (4, 2_000, 13_882_000, "fork", (1, 8)),
    ],
)
def test_smiles_pool_limits_batch_only_cheap_fork_records(
    workers,
    molecules,
    structure_bytes,
    start_method,
    expected,
):
    """Keep batching platform- and complexity-specific with bounded input."""
    assert (
        path_module._smiles_pool_limits(
            workers,
            molecules,
            structure_bytes,
            start_method,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("workers", "chunksize", "start_method", "expected"),
    [
        (2, 64, "fork", True),
        (1, 64, "fork", False),
        (2, 1, "fork", False),
        (2, 64, "spawn", False),
        (2, 64, "forkserver", False),
    ],
)
def test_smiles_structure_reuse_is_limited_to_cheap_fork_batches(
    workers,
    chunksize,
    start_method,
    expected,
):
    """Do not enlarge IPC results on serial, spawn, or complex paths."""
    assert (
        path_module._should_reuse_smiles_worker_structures(
            workers,
            chunksize,
            start_method,
            64 * 1024,
        )
        is expected
    )


def test_smiles_structure_reuse_rejects_unknown_or_large_record_maximums():
    """Do not let a cheap global average hide a large decoded IPC result."""
    assert path_module._should_reuse_smiles_worker_structures(
        2,
        64,
        "fork",
        64 * 1024,
    )
    assert not path_module._should_reuse_smiles_worker_structures(
        2,
        64,
        "fork",
        64 * 1024 + 1,
    )
    assert not path_module._should_reuse_smiles_worker_structures(
        2,
        64,
        "fork",
        None,
    )


@pytest.mark.parametrize(
    (
        "requested_nproc",
        "molecule_count",
        "structure_bytes",
        "start_method",
        "expected_nproc",
    ),
    [
        (8, 5_615, 1_905_017, "spawn", 1),
        (8, 112_300, 38_100_340, "spawn", 1),
        (8, 5_000, 11_995_000, "spawn", 2),
        (8, 2_000, 13_882_000, "spawn", 4),
        (8, 2_000, None, "spawn", 8),
        (16, 5_615, 1_905_017, "fork", 1),
        (16, 112_300, 38_100_340, "fork", 2),
        (16, 280_750, 95_250_850, "fork", 4),
        (8, 5_000, 11_995_000, "fork", 2),
        (8, 2_000, 13_882_000, "fork", 4),
    ],
)
def test_smiles_worker_count_adapts_to_structure_complexity(
    requested_nproc,
    molecule_count,
    structure_bytes,
    start_method,
    expected_nproc,
):
    """Use compressed structure complexity without serializing every molecule."""
    assert (
        path_module._smiles_worker_count(
            requested_nproc,
            molecule_count,
            structure_bytes,
            start_method,
        )
        == expected_nproc
    )


def test_hmm_filter_records_smiles_structure_work(tmp_path):
    """Collect the SMILES scheduling metric while copying retained records."""
    first_structure = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
    )
    second_structure = (
        listtobytes(np.array([0, 1], dtype=np.uint64)),
        listtobytes(np.array([[0, 1]], dtype=np.uint64)),
        listtobytes(np.array([1], dtype=np.uint8)),
    )
    source = tmp_path / "hmm-molecules.bin"
    source.write_bytes(
        b"".join(
            (
                *first_structure,
                listtobytes(np.array([0], dtype=np.uint64)),
                *second_structure,
                listtobytes(np.array([1], dtype=np.uint64)),
            )
        )
    )
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "unused.bond"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=1,
        runHMM=False,
        needprintspecies=False,
    )
    rng.moleculetempfilename = str(source)
    rng.temp1it = 2
    rng.step = 2

    try:
        _HMMFilter(rng).filter()

        assert rng.smilesworkmetrics == hmmfilter_module._SmilesWorkMetrics(
            total_compressed_bytes=sum(
                len(block) for block in (*first_structure, *second_structure)
            ),
            max_compressed_record_bytes=max(
                sum(len(block) for block in first_structure),
                sum(len(block) for block in second_structure),
            ),
        )
    finally:
        for filename in (
            rng.moleculetemp2filename,
            rng.originfilename,
            rng.hmmfilename,
        ):
            if filename:
                Path(filename).unlink(missing_ok=True)


def test_no_hmm_filter_skips_direct_frame_signal_expansion(monkeypatch):
    """Do not materialize an origin signal that matrix will not consume."""
    frame_count = 5_000_000
    frame_block = listtobytes(np.linspace(0, frame_count - 1, 1000, dtype=np.uint64))
    empty = listtobytes([])
    record = (listtobytes(np.array([0], dtype=np.uint64)), empty, empty, frame_block)
    hmm_filter = object.__new__(_HMMFilter)
    hmm_filter.runHMM = False
    hmm_filter.getoriginfile = True
    hmm_filter.step = frame_count

    def fail_frame_decode(_block):
        raise AssertionError("direct no-HMM frames must not expand to an origin signal")

    monkeypatch.setattr(hmmfilter_module, "bytestolist", fail_frame_decode)

    origin_block, hmm_block, returned_record = hmm_filter._getoriginandhmm(record)

    assert origin_block is None
    assert hmm_block is None
    assert returned_record is record


def test_no_hmm_filter_keeps_dense_origin_signal():
    """Retain the bool-signal fallback when uint64 indices would use more memory."""
    frame_count = 1000
    frame_block = listtobytes(np.arange(frame_count, dtype=np.uint64))
    empty = listtobytes([])
    record = (listtobytes(np.array([0], dtype=np.uint64)), empty, empty, frame_block)
    hmm_filter = object.__new__(_HMMFilter)
    hmm_filter.runHMM = False
    hmm_filter.getoriginfile = True
    hmm_filter.step = frame_count

    origin_block, hmm_block, returned_record = hmm_filter._getoriginandhmm(record)

    assert origin_block is not None
    assert hmm_block is None
    assert returned_record is record
    np.testing.assert_array_equal(
        np.asarray(utils_module.bytestolist(origin_block)).reshape((-1,)),
        np.ones(frame_count, dtype=np.bool_),
    )


def test_no_hmm_filter_all_direct_records_avoid_pool(tmp_path, monkeypatch):
    """Keep cheap no-HMM passthrough entirely in the parent process."""
    frame_count = 1000
    empty = listtobytes([])
    records = []
    for atom, frames in (
        (0, np.arange(0, frame_count, 100, dtype=np.uint64)),
        (1, np.arange(50, frame_count, 100, dtype=np.uint64)),
    ):
        records.extend(
            (
                listtobytes(np.array([atom], dtype=np.uint64)),
                empty,
                empty,
                listtobytes(frames),
            )
        )
    source = tmp_path / "direct-no-hmm-molecules.bin"
    source.write_bytes(b"".join(records))
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "unused.bond"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=8,
        runHMM=False,
        needprintspecies=False,
    )
    rng.moleculetempfilename = str(source)
    rng.temp1it = 2
    rng.step = frame_count

    def fail_pool(*_args, **_kwargs):
        raise AssertionError("all-direct no-HMM filtering must not create a Pool")

    monkeypatch.setattr(hmmfilter_module, "run_mp", fail_pool)
    try:
        _HMMFilter(rng).filter()

        assert rng.hmmit == 2
        assert rng.moleculetemp2filename == rng.moleculetempfilename
        assert Path(rng.moleculetemp2filename).read_bytes() == source.read_bytes()
        with open(rng.originfilename, "rb") as handle:
            assert list(utils_module.read_compressed_block(handle)) == []
    finally:
        for filename in (
            rng.moleculetemp2filename,
            rng.originfilename,
            rng.hmmfilename,
        ):
            if filename:
                Path(filename).unlink(missing_ok=True)


def test_no_hmm_filter_lazily_falls_back_for_dense_records(monkeypatch):
    """Start the existing parallel path only when a dense record needs it."""
    frame_count = 1000
    empty = listtobytes([])
    direct_record = (
        listtobytes(np.array([0], dtype=np.uint64)),
        empty,
        empty,
        listtobytes(np.arange(0, frame_count, 100, dtype=np.uint64)),
    )
    dense_record = (
        listtobytes(np.array([1], dtype=np.uint64)),
        empty,
        empty,
        listtobytes(np.arange(frame_count, dtype=np.uint64)),
    )
    hmm_filter = object.__new__(_HMMFilter)
    hmm_filter.runHMM = False
    hmm_filter.getoriginfile = True
    hmm_filter.step = frame_count
    hmm_filter.temp1it = 2
    hmm_filter.nproc = 8
    calls = []

    def tracked_run_mp(nproc, **kwargs):
        calls.append(
            (
                nproc,
                kwargs["total"],
                kwargs["bar"],
                kwargs["chunksize"],
                kwargs["max_inflight"],
                kwargs["unordered"],
            )
        )
        return map(kwargs["func"], kwargs["l"])

    monkeypatch.setattr(hmmfilter_module, "run_mp", tracked_run_mp)
    blocks = iter((*direct_record, *dense_record))

    results = list(hmm_filter._iter_filter_results(blocks))

    assert calls == [(8, 1, False, 100, 1200, False)]
    assert len(results) == 2
    assert results[0][0] is None
    assert (
        np.asarray(utils_module.bytestolist(results[1][0])).reshape((-1,)).size
        == frame_count
    )
    assert hmm_filter._no_hmm_parallel_fallback


def test_hmm_filter_bounds_parallel_record_inflight(monkeypatch):
    """Do not prefetch thousands of full signal records for a large CPU request."""
    hmm_filter = object.__new__(_HMMFilter)
    hmm_filter.runHMM = True
    hmm_filter.nproc = 64
    hmm_filter.temp1it = 10_000
    hmm_filter.step = 1_000_000
    captured = {}

    def tracked_run_mp(nproc, **kwargs):
        captured["nproc"] = nproc
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(hmmfilter_module, "run_mp", tracked_run_mp)

    assert list(hmm_filter._iter_filter_results(iter(()))) == []
    assert captured["nproc"] == 64
    assert captured["chunksize"] == 1
    assert captured["max_inflight"] == 128


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    (
        (1, (100, 9600)),
        (10_000, (100, 9600)),
        (100_000, (10, 1280)),
        (1_000_000, (1, 128)),
    ),
)
def test_hmm_filter_parallel_limits_adapt_to_signal_size(frame_count, expected):
    """Retain small-task batching while bounding full-trajectory signal batches."""
    assert hmmfilter_module._hmm_parallel_pool_limits(64, frame_count) == expected


def test_no_hmm_filter_and_matrix_share_sparse_origin_contract(tmp_path):
    """Store and consume signals only for records that reject direct frames."""
    frame_count = 1000
    direct_frames = np.arange(0, frame_count, 100, dtype=np.uint64)
    dense_frames = np.arange(frame_count, dtype=np.uint64)
    empty = listtobytes([])
    source = tmp_path / "mixed-no-hmm-molecules.bin"
    source.write_bytes(
        b"".join(
            (
                listtobytes(np.array([0], dtype=np.uint64)),
                empty,
                empty,
                listtobytes(direct_frames),
                listtobytes(np.array([1], dtype=np.uint64)),
                empty,
                empty,
                listtobytes(dense_frames),
            )
        )
    )
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "unused.bond"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=1,
        runHMM=False,
        needprintspecies=False,
    )
    rng.moleculetempfilename = str(source)
    rng.temp1it = 2
    rng.step = frame_count

    try:
        _HMMFilter(rng).filter()
        with open(rng.originfilename, "rb") as handle:
            fallback_signals = [
                np.asarray(utils_module.bytestolist(block)).reshape((-1,))
                for block in utils_module.read_compressed_block(handle)
            ]
        assert len(fallback_signals) == 1
        np.testing.assert_array_equal(fallback_signals[0], np.ones(frame_count))

        collector = object.__new__(_CollectSMILESPaths)
        collector.runHMM = False
        collector.originfilename = rng.originfilename
        collector.hmmfilename = None
        collector.moleculetemp2filename = rng.moleculetemp2filename
        collector.hmmit = rng.hmmit
        collector.N = 2
        collector.step = frame_count
        store = collector._getatomeach()
        try:
            expected_direct = np.zeros(frame_count, dtype=np.uint8)
            expected_direct[direct_frames] = 1
            np.testing.assert_array_equal(store.atomeach[0], expected_direct)
            np.testing.assert_array_equal(
                store.atomeach[1],
                np.full(frame_count, 2, dtype=np.uint8),
            )
            assert not np.any(store.conflict)
        finally:
            store.close()
    finally:
        for filename in {
            rng.moleculetempfilename,
            rng.moleculetemp2filename,
            rng.originfilename,
            rng.hmmfilename,
        }:
            if filename:
                Path(filename).unlink(missing_ok=True)


def test_no_hmm_dense_fallback_preserves_signal_molecule_order(
    tmp_path,
    monkeypatch,
):
    """Keep fallback signals aligned when dense workers finish out of order."""
    frame_count = 1000
    direct_frames = np.arange(0, frame_count, 100, dtype=np.uint64)
    first_dense_frames = np.arange(frame_count, dtype=np.uint64)
    second_dense_frames = np.arange(frame_count // 2, dtype=np.uint64)
    empty = listtobytes([])
    source = tmp_path / "ordered-no-hmm-fallback.bin"
    source.write_bytes(
        b"".join(
            b"".join(
                (
                    listtobytes(np.array([atom], dtype=np.uint64)),
                    empty,
                    empty,
                    listtobytes(frames),
                )
            )
            for atom, frames in (
                (0, direct_frames),
                (1, first_dense_frames),
                (2, second_dense_frames),
            )
        )
    )
    monkeypatch.setattr(
        hmmfilter_module,
        "_HMM_TARGET_SIGNAL_VALUES_PER_CHUNK",
        frame_count,
    )
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "unused.bond"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=2,
        runHMM=False,
        needprintspecies=False,
    )
    rng.moleculetempfilename = str(source)
    rng.temp1it = 3
    rng.step = frame_count

    try:
        _DelayedNoHMMFilter(rng).filter()

        collector = object.__new__(_CollectSMILESPaths)
        collector.runHMM = False
        collector.originfilename = rng.originfilename
        collector.hmmfilename = None
        collector.moleculetemp2filename = rng.moleculetemp2filename
        collector.hmmit = rng.hmmit
        collector.N = 3
        collector.step = frame_count
        store = collector._getatomeach()
        try:
            expected_direct = np.zeros(frame_count, dtype=np.uint8)
            expected_direct[direct_frames] = 1
            np.testing.assert_array_equal(store.atomeach[0], expected_direct)
            np.testing.assert_array_equal(
                store.atomeach[1],
                np.full(frame_count, 2, dtype=np.uint8),
            )
            expected_second_dense = np.zeros(frame_count, dtype=np.uint8)
            expected_second_dense[second_dense_frames] = 3
            np.testing.assert_array_equal(store.atomeach[2], expected_second_dense)
            assert not np.any(store.conflict)
        finally:
            store.close()
    finally:
        for filename in {
            rng.moleculetempfilename,
            rng.moleculetemp2filename,
            rng.originfilename,
            rng.hmmfilename,
        }:
            if filename:
                Path(filename).unlink(missing_ok=True)


def test_adaptive_smiles_serial_path_does_not_create_pool(tmp_path, monkeypatch):
    """Run cheap structures in the parent even when more CPUs were requested."""
    structure = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
    )
    molecule_temp = tmp_path / "adaptive-smiles.bin"
    molecule_temp.write_bytes(
        b"".join((*structure, listtobytes(np.array([0], dtype=np.uint64))))
    )
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=8,
    )
    rng.hmmit = 1
    rng.smilesworkmetrics = hmmfilter_module._SmilesWorkMetrics(
        total_compressed_bytes=sum(len(block) for block in structure),
        max_compressed_record_bytes=sum(len(block) for block in structure),
    )
    rng.moleculetemp2filename = str(molecule_temp)
    rng.moleculefilename = str(tmp_path / "adaptive-smiles.txt")
    rng.atomtype = np.array([0])

    def fail_if_run_mp_is_called(*args, **kwargs):
        raise AssertionError("adaptive serial SMILES must not enter run_mp")

    monkeypatch.setattr(path_module, "run_mp", fail_if_run_mp_is_called)
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])
    decode_calls = 0
    original_getatomsandbonds = collector._getatomsandbonds

    def tracked_getatomsandbonds(record):
        nonlocal decode_calls
        decode_calls += 1
        return original_getatomsandbonds(record)

    monkeypatch.setattr(collector, "_getatomsandbonds", tracked_getatomsandbonds)

    collector._printmoleculename(None)

    assert decode_calls == 1
    assert len(collector.mname) == 1
    assert str(collector.mname[0])


def test_smiles_stage_single_process_reads_each_record_once(tmp_path, monkeypatch):
    """Use the parent record directly instead of a Pool plus a second file pass."""
    blocks = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
        listtobytes(np.array([0], dtype=np.uint64)),
    )
    molecule_temp = tmp_path / "molecules.bin"
    molecule_temp.write_bytes(b"".join(blocks))
    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="bond",
        atomname=["H"],
        nproc=1,
    )
    rng.hmmit = 1
    rng.moleculetemp2filename = str(molecule_temp)
    rng.moleculefilename = str(tmp_path / "molecules.txt")
    rng.atomtype = np.array([0])

    def fail_if_run_mp_is_called(*args, **kwargs):
        raise AssertionError("single-process SMILES must not enter run_mp")

    monkeypatch.setattr(path_module, "run_mp", fail_if_run_mp_is_called)
    collector = _CollectSMILESPaths(rng)
    collector.atomnames = np.array(["H"])

    collector._printmoleculename(None)

    assert len(collector.mname) == 1
    assert Path(rng.moleculefilename).read_text().count("\n") == 1


def test_molecule_field_reader_skips_unselected_payloads(tmp_path):
    """Read structure and timeline payloads once through independent handles."""
    blocks = (
        listtobytes(np.array([0], dtype=np.uint64)),
        listtobytes([]),
        listtobytes([]),
        listtobytes(np.arange(1000, dtype=np.uint64)),
    )
    filename = tmp_path / "molecule-blocks.bin"
    filename.write_bytes(b"".join(blocks))

    class TrackingReader:
        def __init__(self, handle):
            self.handle = handle
            self.bytes_read = 0

        def read(self, size):
            data = self.handle.read(size)
            self.bytes_read += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self.handle, name)

    with open(filename, "rb") as structure_handle:
        tracked_structures = TrackingReader(structure_handle)
        structures = list(_iter_molecule_fields(tracked_structures, (0, 1, 2)))
    with open(filename, "rb") as frame_handle:
        tracked_frames = TrackingReader(frame_handle)
        frames = list(_iter_molecule_fields(tracked_frames, (3,)))

    assert structures == [blocks[:3]]
    assert frames == [(blocks[3],)]
    assert tracked_structures.bytes_read == sum(map(len, blocks[:3])) + 64
    assert tracked_frames.bytes_read == len(blocks[3]) + 3 * 64


def test_atom_frame_store_bitpacks_conflict_matrix():
    """Store overlap flags in one bit per atom-frame cell, plus row padding."""
    atom_count = 3
    frame_count = 17
    expected_bytes = atom_count * ((frame_count + 7) // 8)
    store = _AtomFrameStore((atom_count, frame_count), np.uint8)
    try:
        assert os.path.getsize(store.conflict_path) == expected_bytes
        assert store.conflict.nbytes == expected_bytes
    finally:
        store.close()


def test_atom_frame_store_uses_one_byte_per_transition_for_route_index():
    """Keep the shared active-transition index proportional only to frames."""
    store = _AtomFrameStore((3, 17), np.uint8)
    try:
        assert os.path.getsize(store.active_transition_path) == 16
        assert store.active_transitions.nbytes == 16
        assert not np.any(store.active_transitions)
    finally:
        store.close()


def test_packed_conflict_writes_cross_byte_boundaries():
    """Preserve dense and indexed overlap flags across packed-byte edges."""
    store = _AtomFrameStore((3, 17), np.uint8)
    expected = np.zeros(store.shape, dtype=np.bool_)
    try:
        store.atomeach[:] = 0
        store.conflict[:] = False
        path_module._write_dense_atom_frame_span(
            store,
            np.array([0, 1]),
            5,
            14,
            1,
        )
        path_module._write_dense_atom_frame_span(
            store,
            np.array([1, 2]),
            6,
            13,
            2,
        )
        expected[1, 6:13] = True
        path_module._write_indexed_atom_frames(
            store,
            np.array([0, 1, 2]),
            np.array([0, 7, 8, 16]),
            3,
        )
        expected[0, [7, 8]] = True
        expected[1, [7, 8]] = True
        expected[2, [7, 8]] = True
        path_module._write_indexed_atom_frames(
            store,
            np.array([0, 1, 2]),
            np.array([16]),
            4,
        )
        expected[:, 16] = True

        np.testing.assert_array_equal(store.conflict, expected)
        column = store.conflict[:, 8]
        assert not isinstance(column, np.ndarray)
        np.testing.assert_array_equal(column[1:], expected[1:, 8])
        assert not np.any(store.conflict.data[:, -1] & np.uint8(0b11111110))
    finally:
        store.close()


def test_packed_conflict_sparse_write_scans_overlap_once(monkeypatch):
    """Select sparse conflict bits once instead of scanning every atom row."""
    scanned_shapes = []
    original_flatnonzero = packedbool_module.np.flatnonzero

    def tracked_flatnonzero(values):
        scanned_shapes.append(np.asarray(values).shape)
        return original_flatnonzero(values)

    monkeypatch.setattr(packedbool_module.np, "flatnonzero", tracked_flatnonzero)
    store = _AtomFrameStore((1024, 17), np.uint8)
    try:
        store.conflict[:] = False
        overlap = np.zeros((1024, 4), dtype=np.bool_)
        overlap[[0, 0, 100, 500, 1023], [0, 1, 1, 2, 3]] = True
        store.conflict.mark(
            np.arange(1024),
            np.array([0, 7, 8, 16]),
            overlap,
        )

        assert scanned_shapes == [(1024, 4)]
        np.testing.assert_array_equal(store.conflict[:, [0, 7, 8, 16]], overlap)
    finally:
        store.close()


def test_reaction_worker_reads_packed_conflict_columns():
    """Exclude reactions marked by lazily decoded packed conflict columns."""
    store = _AtomFrameStore((2, 2), np.uint8)
    try:
        store.atomeach[:] = [[1, 3], [2, 3]]
        store.conflict[:] = False
        store.conflict.mark(
            np.array([0]),
            np.array([0]),
            np.ones((1, 1), dtype=np.bool_),
        )
        store.save_molecule_names(np.array(["A", "B", "C"]))
        store.flush()
        _initialize_reaction_worker(
            store.atomeach_path,
            store.conflict_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
            store.mname_values_path,
        )

        transition_index, counter = _get_transition_reactions_by_index(0)

        assert transition_index == 0
        assert counter == Counter()
    finally:
        store.close()


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
            store.mname_values_path,
            store.atomtype_path,
            np.array(["H"]),
            ("H",),
            (0, 4),
        )
        route, route_counts, route_payload, modified_atom_events = (
            _get_atom_route_by_index(0)
        )
        np.testing.assert_array_equal(route, [[1, 2]])
        assert route_counts is None
        assert route_payload == b"Atom 1 H: 0 A -> 2 B"
        assert modified_atom_events == 1

        store.atomeach[:, :2] = [[1, 3], [2, 3]]
        store.flush()
        _initialize_reaction_worker(
            store.atomeach_path,
            store.conflict_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
            store.mname_values_path,
        )
        transition_index, counter = _get_transition_reactions_by_index(0)
        assert transition_index == 0
        assert counter == Counter({("A+B", "C"): 1})
    finally:
        store.close()


def test_route_workers_mark_shared_active_transitions():
    """Union exact raw atom changes without returning an event array over IPC."""
    store = _AtomFrameStore((2, 6), np.uint8)
    try:
        store.atomeach[:] = [
            [1, 1, 2, 2, 2, 3],
            [4, 0, 0, 5, 5, 0],
        ]
        store.conflict[:] = False
        store.save_molecule_names(np.array(["A", "B", "C", "D", "E"]))
        store.save_atom_types(np.array([0, 0]))
        store.flush()
        _initialize_route_worker(
            store.atomeach_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
            store.mname_values_path,
            store.atomtype_path,
            np.array(["H"]),
            ("H",),
            (0, 6),
            store.active_transition_path,
            5,
        )

        _get_atom_route_by_index(0)
        _get_atom_route_by_index(1)

        np.testing.assert_array_equal(store.active_transitions, [1, 1, 1, 0, 1])
    finally:
        store.close()


def test_route_worker_compacts_repeated_pairs_before_parent_ipc():
    """Move repeated route aggregation off the serial parent result path."""
    frame_count = 4097
    store = _AtomFrameStore((1, frame_count), np.uint8)
    try:
        store.atomeach[0] = np.resize(
            np.array([1, 2], dtype=np.uint8),
            frame_count,
        )
        store.conflict[:] = False
        store.save_molecule_names(np.array(["A", "B"]))
        store.save_atom_types(np.array([0]))
        store.flush()
        _initialize_route_worker(
            store.atomeach_path,
            store.shape,
            store.molecule_dtype.str,
            store.mname_path,
            store.mname_values_path,
            store.atomtype_path,
            np.array(["H"]),
            ("H",),
            (0, frame_count),
        )

        route_pairs, route_counts, route_payload, modified_atom_events = (
            _get_atom_route_by_index(0)
        )

        np.testing.assert_array_equal(route_pairs, [[1, 2], [2, 1]])
        np.testing.assert_array_equal(route_counts, [2048, 2048])
        assert route_payload.startswith(b"Atom 1 H: 0 A -> 1 B -> 2 A")
        assert modified_atom_events == frame_count - 1
    finally:
        store.close()


def test_reaction_worker_scans_modified_atoms_in_bounded_blocks(monkeypatch):
    """Avoid one full-length intp change index in every reaction worker."""
    atom_count = 100_000
    before = np.concatenate(
        (
            np.ones(atom_count // 2, dtype=np.uint8),
            np.full(atom_count - atom_count // 2, 2, dtype=np.uint8),
        )
    )
    after = np.full(atom_count, 3, dtype=np.uint8)
    conflicts = np.zeros(atom_count, dtype=np.bool_)
    flatnonzero_input_rows = []
    original_flatnonzero = np.flatnonzero

    def tracked_flatnonzero(values):
        flatnonzero_input_rows.append(len(values))
        return original_flatnonzero(values)

    monkeypatch.setattr(reaction_module.np, "flatnonzero", tracked_flatnonzero)

    reactions = _calculate_transition_reactions(
        before,
        after,
        conflicts,
        conflicts,
        np.array(["A", "B", "C"]),
    )

    assert reactions == Counter({("A+B", "C"): 1})
    assert flatnonzero_input_rows
    assert max(flatnonzero_input_rows) <= 65_536


def test_reaction_worker_deduplicates_neighbors_before_dfs(monkeypatch):
    """Do not retain one identical molecule edge for every modified atom."""
    atom_count = 100_000
    captured_neighbor_count = None

    def capture_reaction_graph(reactdict):
        nonlocal captured_neighbor_count
        captured_neighbor_count = sum(
            len(neighbors) for side in reactdict for neighbors in side.values()
        )
        return ()

    monkeypatch.setattr(reaction_module, "dps_reaction", capture_reaction_graph)

    reactions = _calculate_transition_reactions(
        np.ones(atom_count, dtype=np.uint8),
        np.full(atom_count, 2, dtype=np.uint8),
        np.zeros(atom_count, dtype=np.bool_),
        np.zeros(atom_count, dtype=np.bool_),
        np.array(["A", "B"]),
    )

    assert reactions == Counter()
    assert captured_neighbor_count == 2


def test_reaction_worker_compacts_repeated_pairs_before_python_graph(monkeypatch):
    """Do not call the Python graph builder once per repeated changed atom."""
    atom_count = 100_000
    add_calls = 0
    original_add = reaction_module._add_reaction_neighbor

    def tracked_add(*args):
        nonlocal add_calls
        add_calls += 1
        return original_add(*args)

    monkeypatch.setattr(reaction_module, "_add_reaction_neighbor", tracked_add)

    reactions = _calculate_transition_reactions(
        np.ones(atom_count, dtype=np.uint8),
        np.full(atom_count, 2, dtype=np.uint8),
        np.zeros(atom_count, dtype=np.bool_),
        np.zeros(atom_count, dtype=np.bool_),
        np.array(["A", "B"]),
    )

    assert reactions == Counter({("A", "B"): 1})
    maximum_calls = 0
    for block_start in range(0, atom_count, reaction_module._REACTION_ATOM_SCAN_ROWS):
        block_rows = min(
            reaction_module._REACTION_ATOM_SCAN_ROWS,
            atom_count - block_start,
        )
        maximum_calls += 2 * (
            (block_rows + reaction_module._REACTION_PAIR_COMPACTION_BLOCK_ROWS - 1)
            // reaction_module._REACTION_PAIR_COMPACTION_BLOCK_ROWS
        )
    assert add_calls <= maximum_calls


def test_reaction_pair_compaction_reuses_run_detection_buffers(monkeypatch):
    """Allocate run-detection work arrays once, not once per compact block."""
    atom_count = 100_000
    bool_empty_calls = 0
    original_empty = np.empty

    def tracked_empty(shape, *args, **kwargs):
        nonlocal bool_empty_calls
        dtype = kwargs.get("dtype", args[0] if args else None)
        if np.dtype(dtype) == np.dtype(np.bool_) and isinstance(
            shape, (int, np.integer)
        ):
            bool_empty_calls += 1
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(reaction_module.np, "empty", tracked_empty)

    reactions = _calculate_transition_reactions(
        np.ones(atom_count, dtype=np.uint8),
        np.full(atom_count, 2, dtype=np.uint8),
        np.zeros(atom_count, dtype=np.bool_),
        np.zeros(atom_count, dtype=np.bool_),
        np.array(["A", "B"]),
    )

    assert reactions == Counter({("A", "B"): 1})
    assert bool_empty_calls <= 2


def test_reaction_pair_compaction_samples_before_unique_sort(monkeypatch):
    """Keep full-unique changed pairs on the low-allocation Python path."""
    atom_count = 2_000
    observed_rows = []
    original_first_indices = reaction_module._reaction_pair_first_indices

    def tracked_first_indices(before_values, after_values):
        observed_rows.append(len(before_values))
        return original_first_indices(before_values, after_values)

    monkeypatch.setattr(
        reaction_module,
        "_reaction_pair_first_indices",
        tracked_first_indices,
    )
    monkeypatch.setattr(reaction_module, "dps_reaction", lambda reactdict: ())

    reactions = _calculate_transition_reactions(
        np.arange(1, atom_count + 1, dtype=np.uint32),
        np.arange(atom_count + 1, 2 * atom_count + 1, dtype=np.uint32),
        np.zeros(atom_count, dtype=np.bool_),
        np.zeros(atom_count, dtype=np.bool_),
        np.array(["unused"]),
    )

    assert reactions == Counter()
    assert observed_rows
    assert max(observed_rows) <= reaction_module._REACTION_PAIR_COMPACTION_SAMPLE_ROWS


def test_reaction_neighbor_upgrade_preserves_first_seen_order():
    """Keep deterministic DFS ordering when a high-degree list becomes a dict."""
    mapping = {1: []}
    for neighbor in range(10):
        reaction_module._add_reaction_neighbor(mapping, 1, neighbor)
    reaction_module._add_reaction_neighbor(mapping, 1, 3)

    assert isinstance(mapping[1], dict)
    assert list(mapping[1]) == list(range(10))


def test_reaction_pair_first_indices_supports_wide_molecule_ids():
    """Retain first-seen order when pair IDs cannot use the uint32 fast key."""
    before = np.array(
        [2**32 + 2, 2**32 + 1, 2**32 + 2, 2**32 + 3],
        dtype=np.uint64,
    )
    after = np.array(
        [2**32 + 4, 2**32 + 5, 2**32 + 4, 2**32 + 6],
        dtype=np.uint64,
    )

    first_indices = reaction_module._reaction_pair_first_indices(before, after)

    np.testing.assert_array_equal(first_indices, [0, 1, 3])


def test_reaction_neighbor_dedup_matches_duplicate_graph_semantics(monkeypatch):
    """Retain reaction results when one molecule has many repeated neighbors."""
    optimized_add = reaction_module._add_reaction_neighbor
    optimized_should_compact = reaction_module._should_compact_reaction_pairs

    def append_duplicate(mapping, molecule_id, neighbor_id):
        mapping[molecule_id].append(neighbor_id)

    before = np.ones(19 * 32, dtype=np.uint8)
    after = np.tile(np.arange(2, 21, dtype=np.uint8), 32)
    conflicts = np.zeros(len(before), dtype=np.bool_)
    molecule_names = np.array([f"M{index}" for index in range(1, 21)])

    monkeypatch.setattr(reaction_module, "_add_reaction_neighbor", append_duplicate)
    monkeypatch.setattr(
        reaction_module,
        "_should_compact_reaction_pairs",
        lambda *args: False,
    )
    expected = _calculate_transition_reactions(
        before,
        after,
        conflicts,
        conflicts,
        molecule_names,
    )

    monkeypatch.setattr(reaction_module, "_add_reaction_neighbor", optimized_add)
    monkeypatch.setattr(
        reaction_module,
        "_should_compact_reaction_pairs",
        optimized_should_compact,
    )
    actual = _calculate_transition_reactions(
        before,
        after,
        conflicts,
        conflicts,
        molecule_names,
    )

    assert actual == expected
    assert sum(actual.values()) == 1


def test_reaction_pair_compaction_preserves_conflict_semantics(monkeypatch):
    """Keep valid components while excluding a repetitive conflicted component."""
    before = np.concatenate(
        (np.ones(500, dtype=np.uint8), np.full(500, 3, dtype=np.uint8))
    )
    after = np.concatenate(
        (np.full(500, 2, dtype=np.uint8), np.full(500, 4, dtype=np.uint8))
    )
    conflict_before = np.zeros(len(before), dtype=np.bool_)
    conflict_after = np.zeros(len(before), dtype=np.bool_)
    conflict_after[500] = True
    names = np.array(["A", "B", "C", "D"])
    optimized_should_compact = reaction_module._should_compact_reaction_pairs

    monkeypatch.setattr(
        reaction_module,
        "_should_compact_reaction_pairs",
        lambda *args: False,
    )
    expected = _calculate_transition_reactions(
        before,
        after,
        conflict_before,
        conflict_after,
        names,
    )

    monkeypatch.setattr(
        reaction_module,
        "_should_compact_reaction_pairs",
        optimized_should_compact,
    )
    actual = _calculate_transition_reactions(
        before,
        after,
        conflict_before,
        conflict_after,
        names,
    )

    assert expected == Counter({("A", "B"): 1})
    assert actual == expected


def test_reaction_pair_compaction_matches_uncompacted_random_graphs(monkeypatch):
    """Differentially preserve connected reactions across repetitive pair graphs."""
    optimized_should_compact = reaction_module._should_compact_reaction_pairs
    names = np.array([f"M{index}" for index in range(1, 41)])
    conflicts = np.zeros(2_000, dtype=np.bool_)

    for seed in range(20):
        generator = np.random.default_rng(seed)
        before = generator.integers(1, 21, size=2_000, dtype=np.uint16)
        after = generator.integers(21, 41, size=2_000, dtype=np.uint16)

        monkeypatch.setattr(
            reaction_module,
            "_should_compact_reaction_pairs",
            lambda *args: False,
        )
        expected = _calculate_transition_reactions(
            before,
            after,
            conflicts,
            conflicts,
            names,
        )

        monkeypatch.setattr(
            reaction_module,
            "_should_compact_reaction_pairs",
            optimized_should_compact,
        )
        actual = _calculate_transition_reactions(
            before,
            after,
            conflicts,
            conflicts,
            names,
        )

        assert actual == expected


def test_short_reaction_scan_runs_in_parent_without_pool(tmp_path, monkeypatch):
    """Avoid process startup when a reaction stage exposes one tiny transition."""

    def fail_run_mp(*args, **kwargs):
        raise AssertionError("short reaction scan must not create a worker pool")

    monkeypatch.setattr(reaction_module, "run_mp", fail_run_mp)
    store = _AtomFrameStore((1, 2), np.uint8)
    try:
        store.atomeach[0] = [1, 2]
        store.conflict[:] = False
        names = path_module._MoleculeNameTable.from_names(["A", "B"])
        store.save_molecule_names(names)
        store.save_atom_types(np.array([0]))
        store.flush()
        finder = reaction_module.ReactionsFinder(
            SimpleNamespace(
                step=2,
                mname=names,
                reactionabcdfilename=str(tmp_path / "serial.reactionabcd"),
                printreactionevent=False,
                nproc=8,
            )
        )

        finder.findreactions(store)

        assert Path(finder.reactionabcdfilename).read_bytes() == b"1 A->B\n"
    finally:
        store.close()


def test_reaction_scan_uses_route_active_transitions(tmp_path, monkeypatch):
    """Do not scan atom columns for transitions route proved inactive."""
    scanned_transitions = []

    def tracked_transition(transition_index, *_args):
        scanned_transitions.append(int(transition_index))
        return int(transition_index), Counter()

    monkeypatch.setattr(
        reaction_module,
        "_get_transition_reaction_result",
        tracked_transition,
    )
    names = path_module._MoleculeNameTable.from_names(["A"])
    finder = reaction_module.ReactionsFinder(
        SimpleNamespace(
            step=101,
            mname=names,
            reactionabcdfilename=str(tmp_path / "active.reactionabcd"),
            printreactionevent=False,
            nproc=8,
        )
    )
    matrix_store = SimpleNamespace(
        atomeach=object(),
        conflict=object(),
        shape=(10_000, 101),
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
    )
    metrics = SimpleNamespace()

    finder.findreactions(
        matrix_store,
        timed_store=metrics,
        modified_atom_events=2,
        active_transitions=np.array([7, 42], dtype=np.uint8),
    )

    assert scanned_transitions == [7, 42]
    assert metrics.reaction_total_transition_count == 100
    assert metrics.reaction_active_transition_count == 2
    assert metrics.reaction_active_transition_index_available is True
    assert Path(finder.reactionabcdfilename).read_bytes() == b""


def test_parallel_reaction_submits_only_active_transition_indices(
    tmp_path,
    monkeypatch,
):
    """Use sparse original frame indices as the bounded Pool task list."""
    scheduling = {}

    def capture_tasks(nproc, **kwargs):
        scheduling.update(
            nproc=nproc,
            tasks=[int(value) for value in kwargs["l"]],
            total=kwargs["total"],
            chunksize=kwargs["chunksize"],
        )
        return iter(())

    monkeypatch.setattr(reaction_module, "run_mp", capture_tasks)
    monkeypatch.setattr(reaction_module, "_reaction_worker_count", lambda *args: 2)
    names = path_module._MoleculeNameTable.from_names(["A"])
    finder = reaction_module.ReactionsFinder(
        SimpleNamespace(
            step=101,
            mname=names,
            reactionabcdfilename=str(tmp_path / "parallel-active.reactionabcd"),
            printreactionevent=False,
            nproc=8,
        )
    )
    matrix_store = SimpleNamespace(
        atomeach_path="unused-atomeach",
        conflict_path="unused-conflict",
        shape=(10_000, 101),
        molecule_dtype=np.dtype(np.uint16),
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
    )

    finder.findreactions(
        matrix_store,
        modified_atom_events=20_000,
        active_transitions=np.array([7, 42], dtype=np.uint8),
    )

    assert scheduling == {
        "nproc": 2,
        "tasks": [7, 42],
        "total": 2,
        "chunksize": 1,
    }


def test_active_transition_indices_are_validated_and_normalized():
    """Keep sparse task inputs sorted, unique, integral, and in range."""
    np.testing.assert_array_equal(
        reaction_module._normalize_active_transitions([3, 1, 3], 5),
        [1, 3],
    )
    np.testing.assert_array_equal(
        reaction_module._normalize_active_transitions(
            np.array([False, True, False, True, False]),
            5,
        ),
        [1, 3],
    )
    with pytest.raises(ValueError, match="mask length"):
        reaction_module._normalize_active_transitions([True], 5)
    with pytest.raises(ValueError, match="outside"):
        reaction_module._normalize_active_transitions([5], 5)
    with pytest.raises(TypeError, match="integer"):
        reaction_module._normalize_active_transitions([1.5], 5)


def test_zero_active_transitions_skip_all_matrix_columns(tmp_path, monkeypatch):
    """Finish an empty reaction stage without touching atom-frame data."""

    def fail_transition(*_args):
        raise AssertionError("an inactive trajectory must not read a matrix column")

    monkeypatch.setattr(
        reaction_module,
        "_get_transition_reaction_result",
        fail_transition,
    )
    names = path_module._MoleculeNameTable.from_names(["A"])
    finder = reaction_module.ReactionsFinder(
        SimpleNamespace(
            step=1_000_001,
            mname=names,
            reactionabcdfilename=str(tmp_path / "empty.reactionabcd"),
            printreactionevent=False,
            nproc=64,
        )
    )
    matrix_store = SimpleNamespace(
        atomeach=object(),
        conflict=object(),
        shape=(100_000, 1_000_001),
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
    )

    finder.findreactions(
        matrix_store,
        modified_atom_events=0,
        active_transitions=np.zeros(0, dtype=np.int64),
    )

    assert Path(finder.reactionabcdfilename).read_bytes() == b""


def test_sparse_and_full_reaction_transition_scans_match(tmp_path):
    """Skipping unchanged transitions must preserve reaction summaries."""
    store = _AtomFrameStore((4, 7), np.uint8)
    try:
        store.atomeach[:] = [
            [1, 1, 2, 2, 2, 3, 3],
            [4, 4, 5, 5, 5, 6, 6],
            [7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8],
        ]
        store.conflict[:] = False
        names = path_module._MoleculeNameTable.from_names(
            ["A", "B", "C", "D", "E", "F", "G", "H"]
        )
        store.save_molecule_names(names)
        store.flush()
        active_transitions = np.flatnonzero(
            np.any(store.atomeach[:, 1:] != store.atomeach[:, :-1], axis=0)
        )
        common = dict(
            step=7,
            mname=names,
            printreactionevent=False,
            nproc=1,
        )
        full_path = tmp_path / "full.reactionabcd"
        sparse_path = tmp_path / "sparse.reactionabcd"

        reaction_module.ReactionsFinder(
            SimpleNamespace(reactionabcdfilename=str(full_path), **common)
        ).findreactions(store, modified_atom_events=4)
        reaction_module.ReactionsFinder(
            SimpleNamespace(reactionabcdfilename=str(sparse_path), **common)
        ).findreactions(
            store,
            modified_atom_events=4,
            active_transitions=active_transitions,
        )

        assert active_transitions.tolist() == [1, 4]
        assert sparse_path.read_bytes() == full_path.read_bytes()
    finally:
        store.close()


def test_parallel_reaction_summary_uses_deterministic_tie_order(
    tmp_path,
    monkeypatch,
):
    """Keep no-event text stable when workers finish out of transition order."""
    scheduling = {}

    def reversed_results(nproc, **kwargs):
        scheduling.update(
            nproc=nproc,
            chunksize=kwargs["chunksize"],
            max_inflight=kwargs["max_inflight"],
        )
        return iter(
            [
                (1, Counter({("E", "F"): 1, ("A", "B"): 2})),
                (0, Counter({("C", "D"): 1})),
            ]
        )

    monkeypatch.setattr(reaction_module, "run_mp", reversed_results)
    monkeypatch.setattr(reaction_module, "_reaction_worker_count", lambda *args: 2)
    store = _AtomFrameStore((1, 3), np.uint8)
    try:
        names = path_module._MoleculeNameTable.from_names(["A"])
        store.save_molecule_names(names)
        store.flush()
        finder = reaction_module.ReactionsFinder(
            SimpleNamespace(
                step=3,
                mname=names,
                reactionabcdfilename=str(tmp_path / "parallel.reactionabcd"),
                printreactionevent=False,
                nproc=8,
            )
        )

        finder.findreactions(store, modified_atom_events=3)

        assert Path(finder.reactionabcdfilename).read_bytes() == (
            b"2 A->B\n1 C->D\n1 E->F\n"
        )
        assert scheduling == {"nproc": 2, "chunksize": 1, "max_inflight": 4}
    finally:
        store.close()


def test_parallel_reaction_uses_adaptive_chunksize(tmp_path, monkeypatch):
    """Wire cheap long-transition workloads into bounded batched Pool inputs."""
    scheduling = {}

    def empty_results(nproc, **kwargs):
        scheduling.update(
            nproc=nproc,
            chunksize=kwargs["chunksize"],
            max_inflight=kwargs["max_inflight"],
            total=kwargs["total"],
        )
        return iter(())

    monkeypatch.setattr(reaction_module, "run_mp", empty_results)
    monkeypatch.setattr(reaction_module, "_reaction_worker_count", lambda *args: 7)
    names = path_module._MoleculeNameTable.from_names(["A"])
    finder = reaction_module.ReactionsFinder(
        SimpleNamespace(
            step=10_001,
            mname=names,
            reactionabcdfilename=str(tmp_path / "batched.reactionabcd"),
            printreactionevent=False,
            nproc=64,
        )
    )
    matrix_store = SimpleNamespace(
        atomeach_path="unused-atomeach",
        conflict_path="unused-conflict",
        shape=(450, 10_001),
        molecule_dtype=np.dtype(np.uint16),
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
    )

    finder.findreactions(matrix_store, modified_atom_events=360_249)

    assert scheduling == {
        "nproc": 7,
        "chunksize": 32,
        "max_inflight": 448,
        "total": 10_000,
    }
    assert Path(finder.reactionabcdfilename).read_bytes() == b""


@pytest.mark.parametrize(
    (
        "requested_nproc",
        "atom_count",
        "transition_count",
        "modified_atom_events",
        "start_method",
        "expected_nproc",
    ),
    [
        (8, 12_326, 4, 492, "spawn", 1),
        (8, 5_000, 99, 495_000, "spawn", 1),
        (8, 10_000, 499, 4_990_000, "spawn", 4),
        (8, 50_000, 1_999, 1_999, "spawn", 1),
        (64, 500_000, 2_000, 0, "spawn", 2),
        (8, 10_000, 3, None, "spawn", 3),
        (16, 450, 10_000, 360_249, "spawn", 1),
        (16, 450, 10_000, 360_249, "fork", 7),
    ],
)
def test_reaction_worker_count_adapts_to_observed_work(
    requested_nproc,
    atom_count,
    transition_count,
    modified_atom_events,
    start_method,
    expected_nproc,
):
    """Amortize reaction workers over observed changes or very large scans."""
    assert (
        reaction_module._reaction_worker_count(
            requested_nproc,
            atom_count,
            transition_count,
            modified_atom_events,
            start_method,
        )
        == expected_nproc
    )


@pytest.mark.parametrize(
    (
        "worker_count",
        "atom_count",
        "transition_count",
        "modified_atom_events",
        "expected_chunksize",
    ),
    [
        (7, 450, 10_000, 360_249, 32),
        (7, 500_000, 2_000, 0, 2),
        (4, 10_000, 499, 4_990_000, 1),
        (8, 450, 50, 1_800, 1),
        (7, 450, 10_000, None, 1),
    ],
)
def test_reaction_chunksize_adapts_to_batch_work(
    worker_count,
    atom_count,
    transition_count,
    modified_atom_events,
    expected_chunksize,
):
    """Batch cheap transitions without creating long or memory-heavy chunks."""
    assert (
        reaction_module._reaction_chunksize(
            worker_count,
            atom_count,
            transition_count,
            modified_atom_events,
        )
        == expected_chunksize
    )


def test_reaction_worker_block_scan_matches_full_vector_semantics(monkeypatch):
    """Retain reaction counters across scan boundaries and conflict markers."""
    rng = np.random.default_rng(20260807)
    atom_count = 257
    before = rng.integers(1, 10, size=atom_count, dtype=np.uint16)
    after = before.copy()
    changed = rng.random(atom_count) < 0.35
    after[changed] = rng.integers(1, 10, size=np.count_nonzero(changed))
    conflict_before = rng.random(atom_count) < 0.08
    conflict_after = rng.random(atom_count) < 0.08
    molecule_names = np.array([f"Molecule-{index}" for index in range(1, 10)])

    monkeypatch.setattr(reaction_module, "_REACTION_ATOM_SCAN_ROWS", atom_count + 1)
    expected = _calculate_transition_reactions(
        before,
        after,
        conflict_before,
        conflict_after,
        molecule_names,
    )

    monkeypatch.setattr(reaction_module, "_REACTION_ATOM_SCAN_ROWS", 31)
    actual = _calculate_transition_reactions(
        before,
        after,
        conflict_before,
        conflict_after,
        molecule_names,
    )

    assert actual == expected


def test_route_worker_keeps_large_unique_pairs_on_the_fast_path(monkeypatch):
    """Sample unique routes without sorting/copying the full worker result."""
    route_count = 100_000
    left = np.arange(1, route_count + 1, dtype=np.uint64)
    molecule_routes = np.column_stack((left, left + 1))
    unique_input_rows = []
    original_unique = np.unique

    def tracked_unique(values, *args, **kwargs):
        unique_input_rows.append(len(values))
        return original_unique(values, *args, **kwargs)

    monkeypatch.setattr(path_module.np, "unique", tracked_unique)

    route_pairs, route_counts = path_module._compact_route_pairs(molecule_routes)

    assert route_pairs is molecule_routes
    assert route_counts is None
    assert unique_input_rows
    assert max(unique_input_rows) <= path_module._ROUTE_COMPACTION_SAMPLE_ROWS


def test_route_worker_compacts_in_bounded_blocks(monkeypatch):
    """Do not sort an entire long repeated route timeline at once."""
    route_count = 200_000
    molecule_routes = np.resize(
        np.array([[1, 2], [2, 1]], dtype=np.uint64),
        (route_count, 2),
    )
    unique_input_rows = []
    original_unique = np.unique

    def tracked_unique(values, *args, **kwargs):
        unique_input_rows.append(len(values))
        return original_unique(values, *args, **kwargs)

    monkeypatch.setattr(path_module.np, "unique", tracked_unique)

    route_pairs, route_counts = path_module._compact_route_pairs(molecule_routes)

    np.testing.assert_array_equal(route_pairs, [[1, 2], [2, 1]])
    np.testing.assert_array_equal(route_counts, [route_count // 2] * 2)
    assert max(unique_input_rows) <= path_module._ROUTE_COMPACTION_BLOCK_ROWS


def test_route_worker_aborts_compaction_after_unique_pair_cap():
    """Fall back before a near-unique route builds an unbounded worker map."""
    route_count = 70_000
    left = np.arange(100, route_count + 100, dtype=np.uint64)
    molecule_routes = np.column_stack((left, left + 1))
    sample_step = max(
        1,
        route_count // path_module._ROUTE_COMPACTION_SAMPLE_ROWS,
    )
    molecule_routes[::sample_step] = (1, 2)

    route_pairs, route_counts = path_module._compact_route_pairs(molecule_routes)

    assert route_pairs is molecule_routes
    assert route_counts is None


def test_nohmm_route_aggregation_retains_only_species_pair_counts(
    tmp_path,
    monkeypatch,
):
    """Keep repeated route events out of the retained Matrix input."""
    event_count = 100_000
    second_atom_route = np.array([[1, 2], [3, 4]], dtype=np.uint64)

    def fake_run_mp(*args, **kwargs):
        return iter(
            [
                (
                    np.array([[1, 2]], dtype=np.uint64),
                    np.array([event_count], dtype=np.int64),
                    b"Atom 1 H: 0 A -> 1 B",
                    event_count,
                ),
                (
                    second_atom_route,
                    None,
                    b"Atom 2 H: 0 A -> 1 B -> 2 C",
                    2,
                ),
            ]
        )

    monkeypatch.setattr(path_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(path_module, "_route_worker_count", lambda *args: 2)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = False
    collector.step = 2
    collector.atomroutefilename = str(tmp_path / "grouped.route")
    collector.nproc = 1
    collector.N = 2
    collector.atomname = np.array(["H"])
    collector.selectatoms = ["H"]
    collector.mname = path_module._MoleculeNameTable.from_names(["A", "B", "C", "D"])
    matrix_store = SimpleNamespace(
        atomeach_path="unused-atomeach",
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
        atomtype_path="unused-atomtype",
        shape=(2, 2),
        molecule_dtype=np.dtype(np.uint8),
    )

    routes = collector._printatomroute(matrix_store)

    assert isinstance(routes, Counter)
    assert routes == Counter({(0, 1): event_count, (2, 3): 1})
    assert collector._reaction_modified_atom_events == event_count + 2
    assert Path(collector.atomroutefilename).read_text() == (
        "Atom 1 H: 0 A -> 1 B\nAtom 2 H: 0 A -> 1 B -> 2 C\n"
    )


def test_hmm_route_aggregation_counts_each_molecule_pair_once(tmp_path, monkeypatch):
    """Preserve HMM de-duplication while retaining only species counts."""
    second_atom = np.array([[1, 2], [3, 4]], dtype=np.uint64)

    def fake_run_mp(*args, **kwargs):
        return iter(
            [
                (
                    np.array([[1, 2], [2, 3]], dtype=np.uint64),
                    np.array([2, 1], dtype=np.int64),
                    b"Atom 1 H: 0 A -> 1 B",
                    3,
                ),
                (second_atom, None, b"Atom 2 H: 0 A -> 1 B", 2),
            ]
        )

    monkeypatch.setattr(path_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(path_module, "_route_worker_count", lambda *args: 2)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.step = 2
    collector.atomroutefilename = str(tmp_path / "hmm.route")
    collector.nproc = 1
    collector.N = 2
    collector.atomname = np.array(["H"])
    collector.selectatoms = ["H"]
    collector.mname = path_module._MoleculeNameTable.from_names(["A", "B", "A", "C"])
    matrix_store = SimpleNamespace(
        atomeach_path="unused-atomeach",
        mname_path="unused-mname",
        mname_values_path="unused-mname-values",
        atomtype_path="unused-atomtype",
        shape=(2, 2),
        molecule_dtype=np.dtype(np.uint8),
    )

    routes = collector._printatomroute(matrix_store)

    assert routes == Counter({(0, 1): 1, (1, 0): 1, (0, 2): 1})
    assert collector._reaction_modified_atom_events == 5


@pytest.mark.parametrize(
    ("atom_count", "frame_count", "expected_nproc"),
    [(12_326, 5, 1), (1_000, 500, 7), (10_000, 500, 8)],
)
def test_route_worker_count_adapts_to_scan_work(
    atom_count,
    frame_count,
    expected_nproc,
):
    """Do not start more route workers than the timeline scan can keep busy."""
    assert path_module._route_worker_count(8, atom_count, frame_count) == expected_nproc


def test_route_worker_count_avoids_short_high_nproc_tasks():
    """Keep a long narrow trajectory from launching one process per few atoms."""
    assert path_module._route_worker_count(64, 450, 10_001) == 8


def test_short_route_scan_runs_in_parent_without_pool(tmp_path, monkeypatch):
    """Avoid even a one-worker process for a short route timeline."""

    def fail_run_mp(*args, **kwargs):
        raise AssertionError("short route scan must not create a worker pool")

    monkeypatch.setattr(path_module, "run_mp", fail_run_mp)
    store = _AtomFrameStore((1, 2), np.uint8)
    try:
        store.atomeach[0] = [1, 2]
        store.conflict[:] = False
        names = path_module._MoleculeNameTable.from_names(["A", "B"])
        store.save_molecule_names(names)
        store.save_atom_types(np.array([0]))
        store.flush()
        collector = object.__new__(_CollectSMILESPaths)
        collector.runHMM = False
        collector.step = 2
        collector.atomroutefilename = str(tmp_path / "serial.route")
        collector.nproc = 8
        collector.N = 1
        collector.atomname = np.array(["H"])
        collector.atomtype = np.array([0])
        collector.selectatoms = ["H"]
        collector.mname = names

        routes = collector._printatomroute(store)

        assert routes == Counter({(0, 1): 1})
        np.testing.assert_array_equal(collector._reaction_active_transitions, [0])
        assert Path(collector.atomroutefilename).read_bytes() == (
            b"Atom 1 H: 0 A -> 1 B\n"
        )
    finally:
        store.close()


def test_parallel_route_workers_union_active_transitions(tmp_path):
    """Preserve idempotent shared marks when worker writes overlap."""
    atom_count = 256
    frame_count = 1_001
    store = _AtomFrameStore((atom_count, frame_count), np.uint8)
    try:
        store.atomeach[:] = 1
        for atom_index in range(atom_count):
            change_frame = atom_index % 64 + 1
            store.atomeach[atom_index, change_frame:] = 2
        store.conflict[:] = False
        names = path_module._MoleculeNameTable.from_names(["A", "B"])
        store.save_molecule_names(names)
        store.save_atom_types(np.zeros(atom_count, dtype=np.uint8))
        store.flush()
        collector = object.__new__(_CollectSMILESPaths)
        collector.runHMM = False
        collector.step = frame_count
        collector.atomroutefilename = str(tmp_path / "parallel.route")
        collector.nproc = 4
        collector.N = atom_count
        collector.atomname = np.array(["H"])
        collector.atomtype = np.zeros(atom_count, dtype=np.uint8)
        collector.selectatoms = ["H"]
        collector.mname = names

        collector._printatomroute(store)

        np.testing.assert_array_equal(
            collector._reaction_active_transitions,
            np.arange(64),
        )
    finally:
        store.close()


def test_atom_route_avoids_full_intp_nonzero_index(monkeypatch):
    """Filter a dense timeline without allocating one intp index per frame."""
    timeline = np.ones(100_000, dtype=np.uint8)
    original_flatnonzero = np.flatnonzero
    raw_timeline_calls = 0

    def tracked_flatnonzero(values):
        nonlocal raw_timeline_calls
        if values is timeline:
            raw_timeline_calls += 1
        return original_flatnonzero(values)

    monkeypatch.setattr(path_module.np, "flatnonzero", tracked_flatnonzero)

    route, route_text = _calculate_atom_route(
        0,
        timeline,
        np.array([0]),
        np.array(["H"]),
        {"H"},
        np.array(["A"]),
    )

    assert raw_timeline_calls == 0
    assert route.shape == (0, 2)
    assert route_text == "Atom 1 H: 0 A"


def test_route_worker_counts_changes_for_unselected_atoms():
    """Keep reaction scheduling work independent of route display filtering."""
    pairs, counts, payload, modified_atom_events = path_module._get_atom_route_result(
        0,
        np.array([1, 2, 1], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
        np.array(["H"]),
        set(),
        np.array(["A", "B"]),
    )

    assert pairs.shape == (0, 2)
    assert counts is None
    assert payload == b"Atom 1 H: 0 A -> 1 B -> 2 A"
    assert modified_atom_events == 2


def test_route_worker_counts_raw_changes_across_unassigned_frames():
    """Include zero-ID transitions that reaction scans but route text omits."""
    pairs, counts, payload, modified_atom_events = path_module._get_atom_route_result(
        0,
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
        np.array(["H"]),
        {"H"},
        np.array(["A"]),
    )

    assert pairs.shape == (0, 2)
    assert counts is None
    assert payload == b"Atom 1 H: 0 A"
    assert modified_atom_events == 2


def test_raw_route_change_count_uses_bounded_blocks(monkeypatch):
    """Do not allocate a full-timeline comparison just for reaction scheduling."""
    timeline = np.resize(np.array([1, 0], dtype=np.uint8), 200_000)
    compared_rows = []
    original_not_equal = np.not_equal

    def tracked_not_equal(left, right, **kwargs):
        compared_rows.append(len(left))
        return original_not_equal(left, right, **kwargs)

    monkeypatch.setattr(path_module.np, "not_equal", tracked_not_equal)

    change_count = path_module._count_timeline_changes(timeline)

    assert change_count == len(timeline) - 1
    assert compared_rows
    assert max(compared_rows) <= path_module._ROUTE_CHANGE_SCAN_ROWS


def test_atom_route_boolean_scan_matches_previous_semantics():
    """Preserve sparse-timeline route pairs and text while reducing memory."""
    rng = np.random.default_rng(20260807)
    timelines = (
        np.zeros(1000, dtype=np.uint16),
        np.ones(1000, dtype=np.uint16),
        rng.integers(0, 8, size=5000, dtype=np.uint16),
    )
    molecule_names = np.array([f"Molecule-{index}" for index in range(1, 8)])

    for timeline in timelines:
        filtered = timeline[np.flatnonzero(timeline)]
        if filtered.size:
            change_time = np.concatenate(
                (
                    np.zeros(1, dtype=int),
                    np.flatnonzero(np.diff(filtered)) + 1,
                )
            )
            expected_route = filtered[change_time]
        else:
            change_time = np.zeros(0, dtype=int)
            expected_route = np.zeros(0, dtype=timeline.dtype)
        expected_pairs = (
            np.column_stack((expected_route[:-1], expected_route[1:]))
            if expected_route.size
            else np.zeros((0, 2), dtype=int)
        )
        expected_text = "Atom 1 H: " + " -> ".join(
            f"{frame} {name}"
            for frame, name in zip(change_time, molecule_names[expected_route - 1])
        )

        pairs, route_text = _calculate_atom_route(
            0,
            timeline,
            np.array([0]),
            np.array(["H"]),
            {"H"},
            molecule_names,
        )

        np.testing.assert_array_equal(pairs, expected_pairs)
        assert route_text == expected_text


def test_matrix_builder_scans_frame_indices_in_bounded_blocks(
    tmp_path,
    monkeypatch,
):
    """Avoid materializing every nonzero frame index for one molecule."""
    frame_count = 40
    origin = tmp_path / "signals.bin"
    molecules = tmp_path / "molecules.bin"
    signal = np.resize(np.array([True, False], dtype=np.bool_), frame_count)
    signal_block = listtobytes(signal)
    molecule_block = b"".join(
        (
            listtobytes(np.array([0], dtype=np.uint64)),
            listtobytes([]),
            listtobytes([]),
            listtobytes(np.arange(frame_count, dtype=np.uint64)),
        )
    )
    origin.write_bytes(signal_block * 2)
    molecules.write_bytes(molecule_block * 2)
    flatnonzero_input_sizes = []
    matrix_nonzero_calls = 0
    original_flatnonzero = np.flatnonzero
    original_nonzero = np.nonzero

    def tracked_flatnonzero(values):
        flatnonzero_input_sizes.append(np.asarray(values).size)
        return original_flatnonzero(values)

    def tracked_nonzero(values):
        nonlocal matrix_nonzero_calls
        if np.asarray(values).ndim == 2:
            matrix_nonzero_calls += 1
        return original_nonzero(values)

    monkeypatch.setattr(path_module, "_MATRIX_FRAME_SCAN_ROWS", 16, raising=False)
    monkeypatch.setattr(path_module.np, "flatnonzero", tracked_flatnonzero)
    monkeypatch.setattr(path_module.np, "nonzero", tracked_nonzero)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.originfilename = str(origin)
    collector.hmmfilename = str(origin)
    collector.moleculetemp2filename = str(molecules)
    collector.hmmit = 2
    collector.N = 1
    collector.step = frame_count

    store = collector._getatomeach()
    try:
        assert max(flatnonzero_input_sizes) <= 16
        assert matrix_nonzero_calls == 0
        np.testing.assert_array_equal(store.atomeach[0], np.where(signal, 2, 0))
        np.testing.assert_array_equal(store.conflict[0], signal)
    finally:
        store.close()


def test_matrix_builder_avoids_frame_indices_for_dense_blocks(
    tmp_path,
    monkeypatch,
):
    """Write dense molecule spans without allocating one index per frame."""
    frame_count = 40
    origin = tmp_path / "dense-signals.bin"
    molecules = tmp_path / "dense-molecules.bin"
    signal_block = listtobytes(np.ones(frame_count, dtype=np.bool_))
    molecule_block = b"".join(
        (
            listtobytes(np.array([0], dtype=np.uint64)),
            listtobytes([]),
            listtobytes([]),
            listtobytes(np.arange(frame_count, dtype=np.uint64)),
        )
    )
    origin.write_bytes(signal_block * 2)
    molecules.write_bytes(molecule_block * 2)

    def fail_flatnonzero(_values):
        raise AssertionError("dense matrix blocks must not allocate frame indices")

    selected_field_groups = []
    original_iter_molecule_fields = path_module._iter_molecule_fields

    def tracked_iter_molecule_fields(handle, selected_fields, **kwargs):
        selected_field_groups.append(tuple(selected_fields))
        return original_iter_molecule_fields(handle, selected_fields, **kwargs)

    monkeypatch.setattr(path_module, "_MATRIX_FRAME_SCAN_ROWS", 16, raising=False)
    monkeypatch.setattr(path_module.np, "flatnonzero", fail_flatnonzero)
    monkeypatch.setattr(
        path_module,
        "_iter_molecule_fields",
        tracked_iter_molecule_fields,
    )
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.originfilename = str(origin)
    collector.hmmfilename = str(origin)
    collector.moleculetemp2filename = str(molecules)
    collector.hmmit = 2
    collector.N = 1
    collector.step = frame_count

    store = collector._getatomeach()
    try:
        assert selected_field_groups == [(0,)]
        np.testing.assert_array_equal(store.atomeach[0], np.full(frame_count, 2))
        np.testing.assert_array_equal(
            store.conflict[0], np.ones(frame_count, dtype=bool)
        )
    finally:
        store.close()


def test_matrix_builder_does_not_full_scan_empty_blocks_for_density(
    tmp_path,
    monkeypatch,
):
    """Reject empty blocks from a small prefix before a full dense scan."""
    frame_count = 512
    origin = tmp_path / "empty-signals.bin"
    molecules = tmp_path / "empty-molecules.bin"
    origin.write_bytes(listtobytes(np.zeros(frame_count, dtype=np.bool_)))
    empty = listtobytes([])
    molecules.write_bytes(
        b"".join((listtobytes(np.array([0], dtype=np.uint64)), empty, empty, empty))
    )
    all_input_sizes = []
    original_all = np.all

    def tracked_all(values, *args, **kwargs):
        all_input_sizes.append(np.asarray(values).size)
        return original_all(values, *args, **kwargs)

    monkeypatch.setattr(path_module, "_MATRIX_FRAME_SCAN_ROWS", 256, raising=False)
    monkeypatch.setattr(path_module.np, "all", tracked_all)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.originfilename = str(origin)
    collector.hmmfilename = str(origin)
    collector.moleculetemp2filename = str(molecules)
    collector.hmmit = 1
    collector.N = 1
    collector.step = frame_count

    store = collector._getatomeach()
    try:
        assert all_input_sizes
        assert max(all_input_sizes) <= 64
        assert not np.any(store.atomeach)
        assert not np.any(store.conflict)
    finally:
        store.close()


def test_matrix_releases_empty_signal_before_decoding_next(tmp_path, monkeypatch):
    """Do not retain a full signal through its final empty block view."""
    frame_count = 512
    origin = tmp_path / "consecutive-empty-signals.bin"
    molecules = tmp_path / "consecutive-empty-molecules.bin"
    origin.write_bytes(listtobytes(np.zeros(frame_count, dtype=np.bool_)) * 2)
    empty = listtobytes([])
    molecule = b"".join(
        (listtobytes(np.array([0], dtype=np.uint64)), empty, empty, empty)
    )
    molecules.write_bytes(molecule * 2)
    signal_references = []
    original_bytestolist = path_module.bytestolist

    def tracked_bytestolist(block):
        value = original_bytestolist(block)
        array = np.asarray(value)
        if array.dtype == np.bool_ and array.shape == (frame_count,):
            if signal_references:
                assert signal_references[-1]() is None
            signal_references.append(weakref.ref(array))
        return value

    monkeypatch.setattr(path_module, "bytestolist", tracked_bytestolist)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.originfilename = str(origin)
    collector.hmmfilename = str(origin)
    collector.moleculetemp2filename = str(molecules)
    collector.hmmit = 2
    collector.N = 1
    collector.step = frame_count

    store = collector._getatomeach()
    try:
        assert len(signal_references) == 2
        assert not np.any(store.atomeach)
    finally:
        store.close()


def test_matrix_dense_and_sparse_blocks_match_cell_reference(
    tmp_path,
    monkeypatch,
):
    """Preserve mixed block, atom, overwrite, and conflict semantics."""
    signals = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    atoms_by_molecule = (
        np.array([0, 2], dtype=np.uint64),
        np.array([1, 2], dtype=np.uint64),
        np.array([3], dtype=np.uint64),
    )
    origin = tmp_path / "mixed-signals.bin"
    molecules = tmp_path / "mixed-molecules.bin"
    origin.write_bytes(b"".join(listtobytes(signal) for signal in signals))
    empty = listtobytes([])
    molecules.write_bytes(
        b"".join(
            b"".join((listtobytes(atoms), empty, empty, empty))
            for atoms in atoms_by_molecule
        )
    )
    expected_atomeach = np.zeros((4, signals.shape[1]), dtype=np.uint8)
    expected_conflict = np.zeros_like(expected_atomeach, dtype=np.bool_)
    for molecule_id, (signal, atoms) in enumerate(
        zip(signals, atoms_by_molecule),
        start=1,
    ):
        for atom in atoms:
            for frame in np.flatnonzero(signal):
                atom_index = int(atom)
                if expected_atomeach[atom_index, frame] != 0:
                    expected_conflict[atom_index, frame] = True
                expected_atomeach[atom_index, frame] = molecule_id

    monkeypatch.setattr(path_module, "_MATRIX_FRAME_SCAN_ROWS", 4, raising=False)
    monkeypatch.setattr(path_module, "_MATRIX_WRITE_CELLS", 3, raising=False)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = True
    collector.originfilename = str(origin)
    collector.hmmfilename = str(origin)
    collector.moleculetemp2filename = str(molecules)
    collector.hmmit = len(signals)
    collector.N = len(expected_atomeach)
    collector.step = signals.shape[1]

    store = collector._getatomeach()
    try:
        np.testing.assert_array_equal(store.atomeach, expected_atomeach)
        np.testing.assert_array_equal(store.conflict, expected_conflict)
    finally:
        store.close()


def test_no_hmm_matrix_uses_existing_molecule_frame_blocks(tmp_path, monkeypatch):
    """Avoid rebuilding no-HMM frame indices from full-length signals."""
    empty = listtobytes([])
    molecule_file = tmp_path / "no-hmm-molecule-frames.bin"
    molecule_file.write_bytes(
        b"".join(
            (
                listtobytes(np.array([0], dtype=np.uint64)),
                empty,
                empty,
                listtobytes(np.array([0, 2, 4], dtype=np.uint64)),
                listtobytes(np.array([0], dtype=np.uint64)),
                empty,
                empty,
                listtobytes(np.array([1, 2, 3], dtype=np.uint64)),
            )
        )
    )
    invalid_payload = b"not-an-lz4-frame"
    invalid_block = (
        len(invalid_payload).to_bytes(64, byteorder="little") + invalid_payload
    )
    origin_file = tmp_path / "must-not-be-decoded.bin"
    origin_file.write_bytes(invalid_block * 2)
    collector = object.__new__(_CollectSMILESPaths)
    collector.runHMM = False
    collector.originfilename = str(origin_file)
    collector.hmmfilename = str(tmp_path / "must-not-be-opened-hmm.bin")
    collector.moleculetemp2filename = str(molecule_file)
    collector.hmmit = 2
    collector.N = 1
    collector.step = 5
    selected_field_groups = []
    original_iter_molecule_fields = path_module._iter_molecule_fields

    def tracked_iter_molecule_fields(handle, selected_fields, **kwargs):
        selected_field_groups.append(tuple(selected_fields))
        return original_iter_molecule_fields(handle, selected_fields, **kwargs)

    monkeypatch.setattr(
        path_module,
        "_iter_molecule_fields",
        tracked_iter_molecule_fields,
    )

    store = collector._getatomeach()
    try:
        assert selected_field_groups == [(0, 3)]
        np.testing.assert_array_equal(store.atomeach[0], [1, 2, 2, 2, 1])
        np.testing.assert_array_equal(
            store.conflict[0], [False, False, True, False, False]
        )
    finally:
        store.close()


def test_no_hmm_direct_frames_are_signal_memory_bounded():
    """Use stored frame indices only when their decoded size stays bounded."""
    frame_count = 1000
    sparse_frames = listtobytes(np.arange(0, frame_count, 100, dtype=np.uint64))
    dense_frames = listtobytes(np.arange(frame_count, dtype=np.uint64))

    assert path_module._use_direct_no_hmm_frames(sparse_frames, frame_count)
    assert not path_module._use_direct_no_hmm_frames(dense_frames, frame_count)


def test_no_hmm_adaptive_frames_match_signal_matrix(tmp_path):
    """Match forced signal decoding across sparse and dense molecules."""
    frame_count = 257
    rng = np.random.default_rng(20260807)
    signals = (
        np.zeros(frame_count, dtype=np.bool_),
        np.arange(frame_count) % 101 == 0,
        rng.random(frame_count) < 0.08,
        rng.random(frame_count) < 0.55,
        np.ones(frame_count, dtype=np.bool_),
    )
    atoms_by_molecule = (
        np.array([0], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([1, 2], dtype=np.uint64),
        np.array([2], dtype=np.uint64),
    )
    origin = tmp_path / "adaptive-all-signals.bin"
    fallback_origin = tmp_path / "adaptive-fallback-signals.bin"
    molecules = tmp_path / "adaptive-molecule-frames.bin"
    empty = listtobytes([])
    frame_blocks = [
        listtobytes(np.flatnonzero(signal).astype(np.uint64)) for signal in signals
    ]
    molecules.write_bytes(
        b"".join(
            b"".join((listtobytes(atoms), empty, empty, frames))
            for atoms, frames in zip(atoms_by_molecule, frame_blocks)
        )
    )
    decisions = [
        path_module._use_direct_no_hmm_frames(block, frame_count)
        for block in frame_blocks
    ]
    assert any(decisions)
    assert not all(decisions)
    origin.write_bytes(b"".join(listtobytes(signal) for signal in signals))
    fallback_origin.write_bytes(
        b"".join(
            listtobytes(signal)
            for signal, direct_frames in zip(signals, decisions)
            if not direct_frames
        )
    )

    def build(run_hmm):
        collector = object.__new__(_CollectSMILESPaths)
        collector.runHMM = run_hmm
        collector.originfilename = str(fallback_origin)
        collector.hmmfilename = str(origin)
        collector.moleculetemp2filename = str(molecules)
        collector.hmmit = len(signals)
        collector.N = 3
        collector.step = frame_count
        return collector._getatomeach()

    signal_store = build(True)
    adaptive_store = build(False)
    try:
        np.testing.assert_array_equal(adaptive_store.atomeach, signal_store.atomeach)
        np.testing.assert_array_equal(adaptive_store.conflict, signal_store.conflict)
    finally:
        signal_store.close()
        adaptive_store.close()


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


def test_timed_output_cache_defaults_to_sequential_writer_budget():
    """Avoid retaining a large no-reuse raw chunk cache for every dataset."""
    args = main_parser().parse_args(["-i", "input.bond", "-a", "H"])
    rng = ReacNetGenerator(
        inputfilename="input.bond",
        inputfiletype="bond",
        atomname=["H"],
    )
    command = parm2cmd(
        {
            "inputfilename": "input.bond",
            "inputfiletype": "bond",
            "atomname": ["H"],
        }
    )

    assert args.timedoutputcachemib == 1
    assert rng.timedoutputcachemib == 1
    assert "--timed-output-cache-mib" not in command


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
        nproc=2,
        runHMM=False,
        needprintspecies=False,
        printmoleculetime=True,
        printreactionevent=True,
        timedoutputfilename=str(output),
    )

    rng.run()

    with h5py.File(output, "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert int(handle.attrs["timed_output_cache_mib"]) == 1
        for stage in ("molecule", "matrix", "route", "reaction"):
            assert float(handle.attrs[f"step3_{stage}_seconds"]) >= 0
        assert float(handle.attrs["timed_output_write_seconds"]) >= 0
        assert float(handle.attrs["timed_output_molecule_write_seconds"]) >= 0
        assert float(handle.attrs["timed_output_reaction_write_seconds"]) >= 0
        assert float(handle.attrs["timed_output_write_seconds"]) == pytest.approx(
            float(handle.attrs["timed_output_molecule_write_seconds"])
            + float(handle.attrs["timed_output_reaction_write_seconds"])
        )
        assert int(handle.attrs["reaction_total_transition_count"]) == 0
        assert int(handle.attrs["reaction_active_transition_count"]) == 0
        assert not bool(handle.attrs["reaction_active_transition_index_available"])
        assert len(handle["frames/timestep"]) == 1
        assert len(handle["molecules/molecule_id"]) == 1
        np.testing.assert_array_equal(handle["molecule_ranges/start_frame"][:], [0])
        np.testing.assert_array_equal(handle["molecule_ranges/end_frame"][:], [0])
        assert len(handle["reaction_events/count"]) == 0
