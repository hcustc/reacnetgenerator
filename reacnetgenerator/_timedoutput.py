# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
"""Bounded-memory HDF5 output for time-resolved analysis data."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import __version__

_SCHEMA_VERSION = "1"
_HDF5_CHUNK_ROWS = 4096
_PROVENANCE_BATCH_ROWS = 4096
_WRITE_BATCH_ROWS = 4096
_MOLECULE_RANGE_BATCH_ROWS = 65536
_WRITE_BATCH_BYTES = 64 * 1024**2
_ORPHAN_RETENTION_SECONDS = 24 * 60 * 60
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_STEP3_STAGE_NAMES = ("molecule", "matrix", "route", "reaction")


@dataclass
class _MoleculeBatch:
    """Column buffers that must be flushed and reset atomically."""

    species_names: list[str] = field(default_factory=list)
    molecule_ids: list[int] = field(default_factory=list)
    species_ids: list[int] = field(default_factory=list)
    atom_arrays: list[np.ndarray] = field(default_factory=list)
    atom_offsets: list[int] = field(default_factory=list)
    bond_atom_arrays: list[np.ndarray] = field(default_factory=list)
    bond_order_arrays: list[np.ndarray] = field(default_factory=list)
    bond_offsets: list[int] = field(default_factory=list)
    range_molecule_ids: list[int] = field(default_factory=list)
    range_lengths: list[int] = field(default_factory=list)
    range_starts: list[np.ndarray] = field(default_factory=list)
    range_ends: list[np.ndarray] = field(default_factory=list)
    range_count: int = 0
    byte_count: int = 0

    def has_data(self) -> bool:
        return bool(self.species_names or self.molecule_ids or self.range_count)

    def clear(self) -> None:
        self.species_names.clear()
        self.molecule_ids.clear()
        self.species_ids.clear()
        self.atom_arrays.clear()
        self.atom_offsets.clear()
        self.bond_atom_arrays.clear()
        self.bond_order_arrays.clear()
        self.bond_offsets.clear()
        self.range_molecule_ids.clear()
        self.range_lengths.clear()
        self.range_starts.clear()
        self.range_ends.clear()
        self.range_count = 0
        self.byte_count = 0


@dataclass
class _ReactionBatch:
    """Reaction type, event, and block-index buffers with shared lifetime."""

    reactants: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    type_bytes: int = 0
    transition_arrays: list[np.ndarray] = field(default_factory=list)
    reaction_id_arrays: list[np.ndarray] = field(default_factory=list)
    reaction_count_arrays: list[np.ndarray] = field(default_factory=list)
    block_indices: list[int] = field(default_factory=list)
    block_starts: list[int] = field(default_factory=list)
    block_lengths: list[int] = field(default_factory=list)
    event_count: int = 0
    event_bytes: int = 0

    def has_data(self) -> bool:
        return bool(self.reactants or self.event_count or self.block_indices)

    def clear_types(self) -> None:
        self.reactants.clear()
        self.products.clear()
        self.type_bytes = 0

    def clear_events(self) -> None:
        self.transition_arrays.clear()
        self.reaction_id_arrays.clear()
        self.reaction_count_arrays.clear()
        self.block_indices.clear()
        self.block_starts.clear()
        self.block_lengths.clear()
        self.event_count = 0
        self.event_bytes = 0


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
        self.molecule_write_seconds = 0.0
        self.reaction_write_seconds = 0.0
        self.finalize_seconds = 0.0
        self.step3_stage_seconds = dict.fromkeys(_STEP3_STAGE_NAMES, 0.0)
        self._species_ids: dict[str, int] = {}
        self._reaction_ids: dict[tuple[str, str], int] = {}
        self._reaction_totals: Counter[int] = Counter()
        self._reactions_finalized = not self.reaction_enabled
        transition_count = max(0, len(self.timestep) - 1)
        self.reaction_total_transition_count = transition_count
        self.reaction_active_transition_count = transition_count
        self.reaction_active_transition_index_available = False
        self.reaction_event_row_count = 0
        self.reaction_write_batches = 0
        self._reaction_seen = np.zeros(transition_count, dtype=np.bool_)
        self._reaction_batch = _ReactionBatch()
        self.molecule_count = 0
        self.molecule_range_count = 0
        self.molecule_write_batches = 0
        self.maximum_molecule_batch_range_count = 0
        self.maximum_molecule_batch_definition_count = 0
        self.maximum_molecule_batch_bytes = 0
        self._atom_count = 0
        self._bond_count = 0
        self._molecule_batch = _MoleculeBatch()
        self._datasets: dict[str, h5py.Dataset] = {}

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
            self._cache_dataset_handles()
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
        self._datasets.clear()

    def _cache_dataset_handles(self) -> None:
        """Resolve hot-path datasets once instead of for every molecule."""
        assert self.file is not None
        paths = (
            "species/name",
            "molecules/molecule_id",
            "molecules/species_id",
            "molecules/atom_offsets",
            "molecules/bond_offsets",
            "molecules/atom_ids",
            "molecules/bond_atoms",
            "molecules/bond_order",
            "molecule_ranges/molecule_id",
            "molecule_ranges/start_frame",
            "molecule_ranges/end_frame",
            "reaction_types/reactant",
            "reaction_types/product",
            "reaction_types/total_count",
            "reaction_events/transition_index",
            "reaction_events/reaction_id",
            "reaction_events/count",
            "reaction_events/block_start",
            "reaction_events/block_length",
        )
        self._datasets = {path: self.file[path] for path in paths}

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
            chunks=(_HDF5_CHUNK_ROWS,),
            dtype=dtype,
            **options,
        )

    @staticmethod
    def _create_pairs(group: h5py.Group, name: str, dtype) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(_HDF5_CHUNK_ROWS, 2),
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
        index_chunks = (min(_HDF5_CHUNK_ROWS, max(1, transition_count)),)
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
                "timed_output_cache_mib": self.cache_mib,
                "molecule_enabled": self.molecule_enabled,
                "reaction_enabled": self.reaction_enabled,
            }
        )
        frames = self.file["frames"]
        fill_source_arrays = getattr(self.frame_source, "fill_arrays", None)
        timestep_values = None
        if not isinstance(self.timestep, Mapping):
            try:
                candidate_timesteps = np.asarray(self.timestep)
            except (OverflowError, TypeError, ValueError):
                pass
            else:
                if (
                    candidate_timesteps.ndim == 1
                    and len(candidate_timesteps) == len(self.timestep)
                    and np.issubdtype(candidate_timesteps.dtype, np.integer)
                ):
                    timestep_values = candidate_timesteps
        for start in range(0, len(self.timestep), _PROVENANCE_BATCH_ROWS):
            stop = min(start + _PROVENANCE_BATCH_ROWS, len(self.timestep))
            source_ids = np.empty(stop - start, dtype=np.uint32)
            source_frames = np.empty(stop - start, dtype=np.uint64)
            if callable(fill_source_arrays):
                fill_source_arrays(start, stop, source_ids, source_frames)
            else:
                for offset, frame in enumerate(range(start, stop)):
                    source_id, source_frame = self.frame_source.get(frame, (1, frame))
                    source_ids[offset] = source_id
                    source_frames[offset] = source_frame
            if timestep_values is None:
                timesteps = np.fromiter(
                    (int(self.timestep[frame]) for frame in range(start, stop)),
                    dtype=np.int64,
                    count=stop - start,
                )
            else:
                timesteps = np.asarray(
                    timestep_values[start:stop],
                    dtype=np.int64,
                )
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
        species_id = len(self._species_ids) + 1
        self._species_ids[name] = species_id
        self._molecule_batch.species_names.append(name)
        self._molecule_batch.byte_count += len(name.encode("utf-8"))
        return species_id

    @staticmethod
    def _concatenate(arrays: list[np.ndarray], shape, dtype) -> np.ndarray:
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.empty(shape, dtype=dtype)

    def _molecule_batch_is_full(self) -> bool:
        batch = self._molecule_batch
        return (
            len(batch.molecule_ids) >= _WRITE_BATCH_ROWS
            or batch.range_count >= _MOLECULE_RANGE_BATCH_ROWS
            or batch.byte_count >= _WRITE_BATCH_BYTES
        )

    def _flush_molecule_batch(self) -> None:
        batch = self._molecule_batch
        if not batch.has_data():
            return
        self._append(self._datasets["species/name"], batch.species_names)
        self._append(self._datasets["molecules/molecule_id"], batch.molecule_ids)
        self._append(self._datasets["molecules/species_id"], batch.species_ids)
        self._append(
            self._datasets["molecules/atom_ids"],
            self._concatenate(batch.atom_arrays, (0,), np.uint64),
        )
        self._append(self._datasets["molecules/atom_offsets"], batch.atom_offsets)
        self._append(
            self._datasets["molecules/bond_atoms"],
            self._concatenate(batch.bond_atom_arrays, (0, 2), np.uint64),
        )
        self._append(
            self._datasets["molecules/bond_order"],
            self._concatenate(batch.bond_order_arrays, (0,), np.int16),
        )
        self._append(self._datasets["molecules/bond_offsets"], batch.bond_offsets)
        self._append(
            self._datasets["molecule_ranges/molecule_id"],
            np.repeat(
                np.asarray(batch.range_molecule_ids, dtype=np.uint64),
                batch.range_lengths,
            ),
        )
        self._append(
            self._datasets["molecule_ranges/start_frame"],
            self._concatenate(batch.range_starts, (0,), np.uint64),
        )
        self._append(
            self._datasets["molecule_ranges/end_frame"],
            self._concatenate(batch.range_ends, (0,), np.uint64),
        )

        self.maximum_molecule_batch_range_count = max(
            self.maximum_molecule_batch_range_count,
            batch.range_count,
        )
        self.maximum_molecule_batch_definition_count = max(
            self.maximum_molecule_batch_definition_count,
            len(batch.molecule_ids),
        )
        self.maximum_molecule_batch_bytes = max(
            self.maximum_molecule_batch_bytes,
            batch.byte_count,
        )
        batch.clear()
        self.molecule_write_batches += 1

    def _record_molecule_write(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.molecule_write_seconds += elapsed
        self.write_seconds += elapsed

    def _record_reaction_write(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.reaction_write_seconds += elapsed
        self.write_seconds += elapsed

    def flush_molecules(self) -> None:
        """Write pending molecule definitions and ranges as one bounded batch."""
        if not self.molecule_enabled:
            return
        started = time.perf_counter()
        self._flush_molecule_batch()
        self._record_molecule_write(started)

    def add_molecule(
        self,
        molecule_id: int,
        species: str,
        atoms,
        bonds,
        ranges: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Append one molecule definition and its closed existence ranges."""
        if not self.molecule_enabled:
            return
        assert self.file is not None
        started = time.perf_counter()
        atoms_array = np.asarray(atoms, dtype=np.uint64).reshape((-1,))
        bonds_array = np.asarray(bonds, dtype=np.int64)
        if bonds_array.size == 0:
            bond_count = 0
            bond_bytes = 0
        else:
            bonds_array = bonds_array.reshape((-1, 3))
            bond_count = len(bonds_array)
            bond_bytes = bonds_array.nbytes
        batch = self._molecule_batch
        definition_bytes = atoms_array.nbytes + bond_bytes + 32
        if species not in self._species_ids:
            definition_bytes += len(species.encode("utf-8"))
        if batch.has_data() and (
            len(batch.molecule_ids) >= _WRITE_BATCH_ROWS
            or batch.byte_count + definition_bytes > _WRITE_BATCH_BYTES
        ):
            self._flush_molecule_batch()
        self._atom_count += len(atoms_array)
        self._bond_count += bond_count
        batch.molecule_ids.append(int(molecule_id))
        batch.species_ids.append(self._species_id(species))
        batch.atom_arrays.append(atoms_array)
        batch.atom_offsets.append(self._atom_count)
        if bond_count:
            batch.bond_atom_arrays.append(
                np.asarray(bonds_array[:, :2], dtype=np.uint64)
            )
            batch.bond_order_arrays.append(
                np.asarray(bonds_array[:, 2], dtype=np.int16)
            )
        batch.bond_offsets.append(self._bond_count)
        batch.byte_count += atoms_array.nbytes + bond_bytes + 32
        self.molecule_count += 1
        frame_count = len(self.timestep)

        for range_starts, range_ends in ranges:
            starts = np.asarray(range_starts).reshape((-1,))
            ends = np.asarray(range_ends).reshape((-1,))
            if starts.dtype.kind not in "iu":
                starts = starts.astype(np.int64)
            if ends.dtype.kind not in "iu":
                ends = ends.astype(np.int64)
            if starts.shape != ends.shape:
                raise RuntimeError("Timed output contains mismatched molecule ranges")
            if starts.size == 1:
                start_value = int(starts[0])
                end_value = int(ends[0])
                if (
                    start_value < 0
                    or end_value < 0
                    or start_value > end_value
                    or end_value >= frame_count
                ):
                    raise RuntimeError(
                        "Timed output contains an invalid molecule range"
                    )
            elif starts.size:
                if (starts.dtype.kind == "i" and np.any(starts < 0)) or (
                    ends.dtype.kind == "i" and np.any(ends < 0)
                ):
                    raise RuntimeError(
                        "Timed output contains an invalid molecule range"
                    )
                if np.any(starts > ends) or np.any(ends >= frame_count):
                    raise RuntimeError(
                        "Timed output contains an invalid molecule range"
                    )
            starts = starts.astype(np.uint64, copy=False)
            ends = ends.astype(np.uint64, copy=False)
            offset = 0
            while offset < len(starts):
                if self._molecule_batch_is_full():
                    self._flush_molecule_batch()
                remaining_bytes = max(
                    0,
                    _WRITE_BATCH_BYTES - batch.byte_count,
                )
                bytes_per_range = 3 * np.dtype(np.uint64).itemsize
                if remaining_bytes < bytes_per_range and batch.has_data():
                    self._flush_molecule_batch()
                    continue
                byte_capacity = max(
                    1,
                    remaining_bytes // bytes_per_range,
                )
                row_capacity = _MOLECULE_RANGE_BATCH_ROWS - batch.range_count
                take = min(len(starts) - offset, row_capacity, byte_capacity)
                if take <= 0:
                    self._flush_molecule_batch()
                    continue
                stop = offset + take
                start_values = starts[offset:stop]
                end_values = ends[offset:stop]
                batch.range_molecule_ids.append(int(molecule_id))
                batch.range_lengths.append(take)
                batch.range_starts.append(start_values)
                batch.range_ends.append(end_values)
                batch.range_count += take
                batch.byte_count += (
                    take * np.dtype(np.uint64).itemsize
                    + start_values.nbytes
                    + end_values.nbytes
                )
                self.molecule_range_count += take
                offset = stop
                if self._molecule_batch_is_full():
                    self._flush_molecule_batch()
        if self._molecule_batch_is_full():
            self._flush_molecule_batch()
        self._record_molecule_write(started)

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
        if self._reaction_seen[transition_index]:
            raise RuntimeError("Timed output contains a duplicate transition_index")
        batch = self._reaction_batch
        transition_values = []
        reaction_values = []
        counts = []
        for pair, count in sorted(events.items()):
            count = int(count)
            if count <= 0:
                raise RuntimeError("Timed output contains a non-positive count")
            reaction_id = self._reaction_ids.get(pair)
            if reaction_id is None:
                reaction_type_bytes = (
                    len(pair[0].encode("utf-8"))
                    + len(pair[1].encode("utf-8"))
                    + np.dtype(np.uint64).itemsize
                )
                if batch.has_data() and (
                    len(batch.reactants) >= _WRITE_BATCH_ROWS
                    or batch.event_bytes + batch.type_bytes + reaction_type_bytes
                    > _WRITE_BATCH_BYTES
                ):
                    self._flush_reaction_batches()
                reaction_id = len(self._reaction_ids) + 1
                self._reaction_ids[pair] = reaction_id
                batch.reactants.append(pair[0])
                batch.products.append(pair[1])
                batch.type_bytes += reaction_type_bytes
                if (
                    len(batch.reactants) >= _WRITE_BATCH_ROWS
                    or batch.type_bytes >= _WRITE_BATCH_BYTES
                ):
                    self._flush_reaction_type_batch()
            self._reaction_totals[reaction_id] += count
            transition_values.append(int(transition_index))
            reaction_values.append(reaction_id)
            counts.append(count)
        transition_array = np.asarray(transition_values, dtype=np.uint64)
        reaction_id_array = np.asarray(reaction_values, dtype=np.uint32)
        count_array = np.asarray(counts, dtype=np.uint64)
        self._reaction_seen[transition_index] = True
        batch.block_indices.append(transition_index)
        batch.block_starts.append(self.reaction_event_row_count)
        batch.block_lengths.append(len(counts))
        self.reaction_event_row_count += len(counts)
        bytes_per_event = (
            np.dtype(np.uint64).itemsize
            + np.dtype(np.uint32).itemsize
            + np.dtype(np.uint64).itemsize
        )
        offset = 0
        while offset < len(counts):
            if (
                batch.event_count >= _WRITE_BATCH_ROWS
                or batch.event_bytes + batch.type_bytes >= _WRITE_BATCH_BYTES
            ):
                self._flush_reaction_batches()
            remaining_bytes = max(
                0,
                _WRITE_BATCH_BYTES - batch.event_bytes - batch.type_bytes,
            )
            if remaining_bytes < bytes_per_event and (
                batch.event_count or batch.reactants
            ):
                self._flush_reaction_batches()
                continue
            byte_capacity = max(1, remaining_bytes // bytes_per_event)
            row_capacity = _WRITE_BATCH_ROWS - batch.event_count
            take = min(len(counts) - offset, row_capacity, byte_capacity)
            if take <= 0:
                self._flush_reaction_batches()
                continue
            stop = offset + take
            transition_values_block = transition_array[offset:stop]
            reaction_id_values_block = reaction_id_array[offset:stop]
            count_values_block = count_array[offset:stop]
            batch.transition_arrays.append(transition_values_block)
            batch.reaction_id_arrays.append(reaction_id_values_block)
            batch.reaction_count_arrays.append(count_values_block)
            batch.event_count += take
            batch.event_bytes += (
                transition_values_block.nbytes
                + reaction_id_values_block.nbytes
                + count_values_block.nbytes
            )
            offset = stop
            if (
                batch.event_count >= _WRITE_BATCH_ROWS
                or batch.event_bytes + batch.type_bytes >= _WRITE_BATCH_BYTES
            ):
                self._flush_reaction_batches()
        self._record_reaction_write(started)

    def _flush_reaction_type_batch(self) -> None:
        batch = self._reaction_batch
        if not batch.reactants:
            return
        self._append(self._datasets["reaction_types/reactant"], batch.reactants)
        self._append(self._datasets["reaction_types/product"], batch.products)
        self._append(
            self._datasets["reaction_types/total_count"],
            np.zeros(len(batch.reactants), dtype=np.uint64),
        )
        batch.clear_types()

    def _flush_reaction_event_batch(self) -> None:
        batch = self._reaction_batch
        if not batch.event_count:
            return
        self._append(
            self._datasets["reaction_events/transition_index"],
            self._concatenate(batch.transition_arrays, (0,), np.uint64),
        )
        self._append(
            self._datasets["reaction_events/reaction_id"],
            self._concatenate(batch.reaction_id_arrays, (0,), np.uint32),
        )
        self._append(
            self._datasets["reaction_events/count"],
            self._concatenate(batch.reaction_count_arrays, (0,), np.uint64),
        )
        if batch.block_indices:
            block_indices = np.asarray(batch.block_indices, dtype=np.int64)
            block_order = np.argsort(block_indices)
            block_indices = block_indices[block_order]
            self._datasets["reaction_events/block_start"][block_indices] = np.asarray(
                batch.block_starts,
                dtype=np.uint64,
            )[block_order]
            self._datasets["reaction_events/block_length"][block_indices] = np.asarray(
                batch.block_lengths,
                dtype=np.uint64,
            )[block_order]
        batch.clear_events()
        self.reaction_write_batches += 1

    def _flush_reaction_batches(self) -> None:
        self._flush_reaction_type_batch()
        self._flush_reaction_event_batch()

    def finalize_reactions(self) -> None:
        """Finalize total reaction counts after all transitions were staged."""
        if not self.reaction_enabled:
            return
        assert self.file is not None
        started = time.perf_counter()
        self._flush_reaction_batches()
        totals = self._datasets["reaction_types/total_count"]
        total_values = np.zeros(len(self._reaction_ids), dtype=np.uint64)
        for reaction_id, count in self._reaction_totals.items():
            total_values[reaction_id - 1] = int(count)
        totals[:] = total_values
        self._reactions_finalized = True
        self.file.flush()
        self._record_reaction_write(started)

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
        self.flush_molecules()
        started = time.perf_counter()
        self.status = "complete"
        self.file.attrs.update(
            {
                "status": self.status,
                "completed_at": time.time(),
                "molecule_count": self.molecule_count,
                "molecule_range_count": self.molecule_range_count,
                "molecule_write_batches": self.molecule_write_batches,
                "maximum_molecule_batch_range_count": (
                    self.maximum_molecule_batch_range_count
                ),
                "maximum_molecule_batch_definition_count": (
                    self.maximum_molecule_batch_definition_count
                ),
                "maximum_molecule_batch_bytes": self.maximum_molecule_batch_bytes,
                "molecule_range_batch_row_limit": _MOLECULE_RANGE_BATCH_ROWS,
                "timed_output_write_batch_byte_limit": _WRITE_BATCH_BYTES,
                "reaction_type_count": len(self._reaction_ids),
                "reaction_event_row_count": self.reaction_event_row_count,
                "reaction_write_batches": self.reaction_write_batches,
                "reaction_total_transition_count": (
                    self.reaction_total_transition_count
                ),
                "reaction_active_transition_count": (
                    self.reaction_active_transition_count
                ),
                "reaction_active_transition_index_available": (
                    self.reaction_active_transition_index_available
                ),
                "timed_output_write_seconds": self.write_seconds,
                "timed_output_molecule_write_seconds": (self.molecule_write_seconds),
                "timed_output_reaction_write_seconds": (self.reaction_write_seconds),
                **{
                    f"step3_{name}_seconds": self.step3_stage_seconds[name]
                    for name in _STEP3_STAGE_NAMES
                },
            }
        )
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
