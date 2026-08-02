# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
"""Bounded-memory HDF5 output for time-resolved analysis data."""

from __future__ import annotations

import itertools
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from . import __version__

_SCHEMA_VERSION = "1"
_APPEND_ROWS = 4096
_ORPHAN_RETENTION_SECONDS = 24 * 60 * 60
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


class _FileLock:
    """Exclusive advisory lock for one published output path."""

    def __init__(self, path: str):
        self.path = path
        self.handle = None
        self.acquired = False

    def acquire(self) -> bool:
        self.handle = open(self.path, "a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":  # pragma: no cover - Windows CI only
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported platform
                self.handle.close()
                self.handle = None
                return False
        except (BlockingIOError, OSError):
            self.handle.close()
            self.handle = None
            return False
        self.acquired = True
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        if self.acquired:
            try:
                self.handle.seek(0)
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                elif os.name == "nt":  # pragma: no cover - Windows CI only
                    import msvcrt

                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        self.handle.close()
        self.handle = None
        self.acquired = False


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: str) -> None:
    if os.name != "posix":  # pragma: no cover - no portable directory fsync
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class TimedOutputStore:
    """Stream normalized molecule ranges and reaction events to HDF5."""

    def __init__(
        self,
        filename: str,
        *,
        cache_mib: int,
        input_filenames: list[str],
        timestep: dict[int, int],
        frame_source: dict[int, tuple[int, int]] | None,
        stepinterval: int,
        molecule_enabled: bool,
        reaction_enabled: bool,
    ):
        self.filename = os.path.abspath(filename)
        self.output_dir = os.path.dirname(self.filename)
        self.cache_mib = int(cache_mib)
        if self.cache_mib <= 0:
            raise ValueError("timedoutputcachemib must be a positive integer")
        self.input_filenames = [str(path) for path in input_filenames]
        self.timestep = timestep
        self.frame_source = frame_source or {}
        self.stepinterval = int(stepinterval)
        self.molecule_enabled = bool(molecule_enabled)
        self.reaction_enabled = bool(reaction_enabled)
        self.job_id = uuid.uuid4().hex
        self.temp_filename = f"{self.filename}.tmp.{self.job_id}"
        self.lock_filename = f"{self.filename}.lock"
        self.lock = _FileLock(self.lock_filename)
        self.file: h5py.File | None = None
        self.status = "building"
        self.published = False
        self.write_seconds = 0.0
        self.finalize_seconds = 0.0
        self._species_ids: dict[str, int] = {}
        self._reaction_ids: dict[tuple[str, str], int] = {}
        self._reaction_totals: Counter[int] = Counter()
        self._reactions_finalized = not self.reaction_enabled

    def __enter__(self) -> TimedOutputStore:
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.lock.acquire():
            raise RuntimeError(f"Timed output is already being built: {self.filename}")
        try:
            self.cleanup_orphans(self.filename, lock_held=True)
            self.file = h5py.File(
                self.temp_filename,
                "w",
                rdcc_nbytes=self.cache_mib * 1024**2,
            )
            self._create_schema()
            self._insert_provenance()
            self.file.flush()
        except BaseException:
            self.mark_failed()
            self._close_file()
            self._release_lock()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.published:
            self.mark_failed()
        self._close_file()
        self._release_lock()

    @staticmethod
    def cleanup_orphans(
        filename: str,
        retention_seconds: float = _ORPHAN_RETENTION_SECONDS,
        *,
        lock_held: bool = False,
    ) -> None:
        """Remove stale temporary HDF5 artifacts for one unlocked target."""
        final_path = Path(os.path.abspath(filename))
        cleanup_lock = None
        if not lock_held:
            cleanup_lock = _FileLock(f"{final_path}.lock")
            if not cleanup_lock.acquire():
                return
        try:
            cutoff = time.time() - float(retention_seconds)
            for candidate in final_path.parent.glob(f"{final_path.name}.tmp.*"):
                try:
                    if candidate.stat().st_mtime < cutoff:
                        candidate.unlink()
                except FileNotFoundError:
                    pass
        finally:
            if cleanup_lock is not None:
                cleanup_lock.release()
                try:
                    os.unlink(cleanup_lock.path)
                except FileNotFoundError:
                    pass

    def _release_lock(self) -> None:
        self.lock.release()
        try:
            os.unlink(self.lock_filename)
        except FileNotFoundError:
            pass

    def _close_file(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    @staticmethod
    def _create_vector(
        group: h5py.Group,
        name: str,
        dtype,
        *,
        compression: bool = True,
    ) -> h5py.Dataset:
        options = {}
        if compression and dtype != _STRING_DTYPE:
            options = {
                "compression": "gzip",
                "compression_opts": 1,
                "shuffle": True,
            }
        return group.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(_APPEND_ROWS,),
            dtype=dtype,
            **options,
        )

    @staticmethod
    def _create_pairs(group: h5py.Group, name: str, dtype) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(_APPEND_ROWS, 2),
            dtype=dtype,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

    def _create_schema(self) -> None:
        assert self.file is not None
        sources = self.file.create_group("sources")
        sources.create_dataset(
            "path",
            data=np.asarray(self.input_filenames, dtype=object),
            dtype=_STRING_DTYPE,
        )
        sources.create_dataset(
            "ordinal",
            data=np.arange(len(self.input_filenames), dtype=np.uint32),
        )

        frames = self.file.create_group("frames")
        self._create_vector(frames, "source_id", np.uint32)
        self._create_vector(frames, "source_frame", np.uint64)
        self._create_vector(frames, "timestep", np.int64)

        species = self.file.create_group("species")
        self._create_vector(species, "name", _STRING_DTYPE, compression=False)

        molecules = self.file.create_group("molecules")
        self._create_vector(molecules, "molecule_id", np.uint64)
        self._create_vector(molecules, "species_id", np.uint32)
        atom_offsets = self._create_vector(molecules, "atom_offsets", np.uint64)
        bond_offsets = self._create_vector(molecules, "bond_offsets", np.uint64)
        atom_offsets.resize((1,))
        atom_offsets[0] = 0
        bond_offsets.resize((1,))
        bond_offsets[0] = 0
        self._create_vector(molecules, "atom_ids", np.uint64)
        self._create_pairs(molecules, "bond_atoms", np.uint64)
        self._create_vector(molecules, "bond_order", np.int16)

        ranges = self.file.create_group("molecule_ranges")
        self._create_vector(ranges, "molecule_id", np.uint64)
        self._create_vector(ranges, "start_frame", np.uint64)
        self._create_vector(ranges, "end_frame", np.uint64)

        reaction_types = self.file.create_group("reaction_types")
        self._create_vector(
            reaction_types, "reactant", _STRING_DTYPE, compression=False
        )
        self._create_vector(reaction_types, "product", _STRING_DTYPE, compression=False)
        self._create_vector(reaction_types, "total_count", np.uint64)

        events = self.file.create_group("reaction_events")
        self._create_vector(events, "transition_index", np.uint64)
        self._create_vector(events, "reaction_id", np.uint32)
        self._create_vector(events, "count", np.uint64)
        transition_count = max(0, len(self.timestep) - 1)
        index_chunks = (min(_APPEND_ROWS, max(1, transition_count)),)
        for name in ("block_start", "block_length"):
            events.create_dataset(
                name,
                shape=(transition_count,),
                maxshape=(None,),
                dtype=np.uint64,
                chunks=index_chunks,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
                fillvalue=0,
            )

    def _insert_provenance(self) -> None:
        assert self.file is not None
        now = time.time()
        self.file.attrs.update(
            {
                "schema_version": _SCHEMA_VERSION,
                "reacnetgenerator_version": __version__,
                "status": self.status,
                "job_id": self.job_id,
                "started_at": now,
                "stepinterval": self.stepinterval,
                "frame_count": len(self.timestep),
                "source_order": json.dumps(self.input_filenames),
                "molecule_enabled": self.molecule_enabled,
                "reaction_enabled": self.reaction_enabled,
            }
        )
        frames = self.file["frames"]
        for start in range(0, len(self.timestep), _APPEND_ROWS):
            stop = min(start + _APPEND_ROWS, len(self.timestep))
            source_ids = np.empty(stop - start, dtype=np.uint32)
            source_frames = np.empty(stop - start, dtype=np.uint64)
            timesteps = np.empty(stop - start, dtype=np.int64)
            for offset, frame in enumerate(range(start, stop)):
                source_id, source_frame = self.frame_source.get(frame, (1, frame))
                source_ids[offset] = source_id
                source_frames[offset] = source_frame
                timesteps[offset] = int(self.timestep[frame])
            self._append(frames["source_id"], source_ids)
            self._append(frames["source_frame"], source_frames)
            self._append(frames["timestep"], timesteps)

    @staticmethod
    def _append(dataset: h5py.Dataset, values) -> None:
        values = np.asarray(values, dtype=dataset.dtype)
        if values.size == 0:
            return
        if dataset.ndim == 1:
            values = values.reshape((-1,))
        else:
            values = values.reshape((-1, *dataset.shape[1:]))
        start = dataset.shape[0]
        dataset.resize((start + values.shape[0], *dataset.shape[1:]))
        dataset[start:] = values

    def _species_id(self, name: str) -> int:
        species_id = self._species_ids.get(name)
        if species_id is not None:
            return species_id
        assert self.file is not None
        species_id = len(self._species_ids) + 1
        self._species_ids[name] = species_id
        self._append(self.file["species/name"], [name])
        return species_id

    def add_molecule(
        self,
        molecule_id: int,
        species: str,
        atoms,
        bonds,
        ranges: Iterable[tuple[int, int]],
    ) -> None:
        """Append one molecule definition and its closed existence ranges."""
        if not self.molecule_enabled:
            return
        assert self.file is not None
        started = time.perf_counter()
        atoms_array = np.asarray(atoms, dtype=np.uint64).reshape((-1,))
        bonds_array = np.asarray(bonds, dtype=np.int64)
        if bonds_array.size == 0:
            bonds_array = np.empty((0, 3), dtype=np.int64)
        else:
            bonds_array = bonds_array.reshape((-1, 3))
        molecules = self.file["molecules"]
        self._append(molecules["molecule_id"], [int(molecule_id)])
        self._append(molecules["species_id"], [self._species_id(species)])
        self._append(molecules["atom_ids"], atoms_array)
        self._append(
            molecules["atom_offsets"],
            [int(molecules["atom_offsets"][-1]) + len(atoms_array)],
        )
        self._append(molecules["bond_atoms"], bonds_array[:, :2])
        self._append(molecules["bond_order"], bonds_array[:, 2])
        self._append(
            molecules["bond_offsets"],
            [int(molecules["bond_offsets"][-1]) + len(bonds_array)],
        )

        range_iterator = iter(ranges)
        while True:
            batch = list(itertools.islice(range_iterator, _APPEND_ROWS))
            if not batch:
                break
            starts = np.fromiter((int(x[0]) for x in batch), dtype=np.uint64)
            ends = np.fromiter((int(x[1]) for x in batch), dtype=np.uint64)
            if np.any(starts > ends) or np.any(ends >= len(self.timestep)):
                raise RuntimeError("Timed output contains an invalid molecule range")
            ranges_group = self.file["molecule_ranges"]
            self._append(
                ranges_group["molecule_id"],
                np.full(len(batch), int(molecule_id), dtype=np.uint64),
            )
            self._append(ranges_group["start_frame"], starts)
            self._append(ranges_group["end_frame"], ends)
        self.write_seconds += time.perf_counter() - started

    def stage_reaction_events(
        self,
        transition_index: int,
        events: Counter[tuple[str, str]],
    ) -> None:
        """Append one transition's already-aggregated reaction events."""
        if not self.reaction_enabled or not events:
            return
        if transition_index < 0 or transition_index >= max(0, len(self.timestep) - 1):
            raise RuntimeError("Timed output contains an invalid transition_index")
        assert self.file is not None
        started = time.perf_counter()
        event_group = self.file["reaction_events"]
        if int(event_group["block_length"][transition_index]) != 0:
            raise RuntimeError("Timed output contains a duplicate transition_index")
        transition_values = []
        reaction_values = []
        counts = []
        types = self.file["reaction_types"]
        for pair, count in sorted(events.items()):
            count = int(count)
            if count <= 0:
                raise RuntimeError("Timed output contains a non-positive count")
            reaction_id = self._reaction_ids.get(pair)
            if reaction_id is None:
                reaction_id = len(self._reaction_ids) + 1
                self._reaction_ids[pair] = reaction_id
                self._append(types["reactant"], [pair[0]])
                self._append(types["product"], [pair[1]])
                self._append(types["total_count"], [0])
            self._reaction_totals[reaction_id] += count
            transition_values.append(int(transition_index))
            reaction_values.append(reaction_id)
            counts.append(count)
        block_start = len(event_group["count"])
        self._append(event_group["transition_index"], transition_values)
        self._append(event_group["reaction_id"], reaction_values)
        self._append(event_group["count"], counts)
        event_group["block_start"][transition_index] = block_start
        event_group["block_length"][transition_index] = len(counts)
        self.write_seconds += time.perf_counter() - started

    def finalize_reactions(self) -> None:
        """Finalize total reaction counts after all transitions were staged."""
        if not self.reaction_enabled:
            return
        assert self.file is not None
        totals = self.file["reaction_types/total_count"]
        for reaction_id, count in self._reaction_totals.items():
            totals[reaction_id - 1] = int(count)
        self._reactions_finalized = True
        self.file.flush()

    def iter_reaction_summaries(self):
        """Yield reaction summaries ordered by count and names."""
        assert self.file is not None
        reactants = self.file["reaction_types/reactant"][:]
        products = self.file["reaction_types/product"][:]
        totals = self.file["reaction_types/total_count"][:]
        rows = (
            (int(total), _decode_string(reactant), _decode_string(product))
            for total, reactant, product in zip(totals, reactants, products)
        )
        yield from sorted(rows, key=lambda row: (-row[0], row[1], row[2]))

    def finalize_and_publish(self) -> None:
        """Validate, close, fsync, and atomically publish the HDF5 artifact."""
        assert self.file is not None
        if not self._reactions_finalized:
            raise RuntimeError("Reaction events have not been finalized")
        started = time.perf_counter()
        self.status = "complete"
        self.file.attrs["status"] = self.status
        self.file.attrs["completed_at"] = time.time()
        self.file.flush()
        self._close_file()
        _fsync_file(self.temp_filename)
        os.replace(self.temp_filename, self.filename)
        _fsync_directory(self.output_dir)
        self.finalize_seconds += time.perf_counter() - started
        self.published = True
        self._release_lock()

    def mark_failed(self) -> None:
        """Mark an unpublished temporary HDF5 file as failed."""
        self.status = "failed"
        if self.file is None:
            return
        try:
            self.file.attrs["status"] = self.status
            self.file.attrs["failed_at"] = time.time()
            self.file.flush()
        except (OSError, RuntimeError, ValueError):
            pass
