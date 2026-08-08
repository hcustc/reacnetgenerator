# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Generate Matrix.

A reaction network cannot accommodate too many species, so only the first
species which have the most reactions are taken. A reaction matrix can be
generated.
"""

import operator
import os
import tempfile
from collections import Counter

import numpy as np
import pandas as pd

from ._logging import logger
from ._moleculenames import _MoleculeNameTable, _unsigned_dtype_for_maximum
from .utils import (
    SharedRNGData,
    WriteBuffer,
    _iter_compressed_record_fields,
    bytestolist,
)

_SPECIES_COUNT_BLOCK_BYTES = 64 * 1024 * 1024
_SPECIES_EVENT_CHUNK_ROWS = 65536
_SPECIES_OBSERVATION_SPOOL_BYTES = 256 * 1024 * 1024


class _SpeciesTimelineStore:
    """Aggregate species counts with bounded RAM and adaptive temporary storage."""

    def __init__(self, frame_count, species_count, molecule_count):
        self.frame_count = int(frame_count)
        self.species_count = int(species_count)
        self.molecule_count = int(molecule_count)
        if min(self.frame_count, self.species_count, self.molecule_count) < 0:
            raise ValueError("Species timeline dimensions must be non-negative")
        self.count_dtype = _unsigned_dtype_for_maximum(self.molecule_count)
        self.cell_dtype = np.dtype(
            [
                ("count", self.count_dtype),
                ("priority", self.count_dtype),
            ]
        )
        count_row_bytes = max(1, self.species_count * self.cell_dtype.itemsize)
        self.block_frame_count = max(
            1,
            _SPECIES_COUNT_BLOCK_BYTES // count_row_bytes,
        )
        self.block_count = (
            self.frame_count + self.block_frame_count - 1
        ) // self.block_frame_count
        self.block_rows = np.zeros(self.block_count, dtype=np.uint64)
        self.total_rows = 0
        self.added_rows = 0
        self.mode = None
        self.path = None
        self.data = None
        self.block_offsets = None
        self.block_cursors = None
        self.frame_dtype = _unsigned_dtype_for_maximum(max(0, self.frame_count - 1))
        self.species_dtype = _unsigned_dtype_for_maximum(max(0, self.species_count - 1))
        self.event_dtype = np.dtype(
            [
                ("frame", self.frame_dtype),
                ("species", self.species_dtype),
            ]
        )
        self.observation_dtype = np.dtype(
            [
                ("frame", self.frame_dtype),
                ("species", self.species_dtype),
                ("priority", self.count_dtype),
            ]
        )
        self.observation_path = None
        self.observation_file = None
        self.observation_buffer = None
        self.observation_buffer_rows = 0
        self.observation_rows = 0
        self.observation_complete = True
        self.observation_reused = False
        self.peak_temporary_bytes = 0
        self.dense_bytes = (
            self.frame_count * self.species_count * self.cell_dtype.itemsize
        )
        if (
            self.frame_count > 0
            and self.species_count > 0
            and self.dense_bytes <= _SPECIES_COUNT_BLOCK_BYTES
        ):
            self._allocate_dense()
        else:
            self._create_observation_spool()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _frame_array(self, frames):
        frames = np.asarray(frames)
        if frames.ndim != 1:
            frames = frames.reshape((-1,))
        if not np.issubdtype(frames.dtype, np.integer):
            raise TypeError("Molecule frame indices must use an integer dtype")
        if frames.size:
            if frames.size == 1:
                minimum = maximum = int(frames[0])
            else:
                minimum = int(frames.min())
                maximum = int(frames.max())
            if minimum < 0 or maximum >= self.frame_count:
                raise IndexError("Molecule frame index is outside the trajectory")
        return frames

    @staticmethod
    def _chunks(values):
        for start in range(0, len(values), _SPECIES_EVENT_CHUNK_ROWS):
            yield values[start : start + _SPECIES_EVENT_CHUNK_ROWS]

    def _create_observation_spool(self):
        handle, self.observation_path = tempfile.mkstemp(
            prefix="reacnetgenerator-species-observed-",
            suffix=".bin",
        )
        self.observation_file = os.fdopen(handle, "w+b")
        self.observation_buffer = np.empty(
            _SPECIES_EVENT_CHUNK_ROWS,
            dtype=self.observation_dtype,
        )

    def _flush_observation_buffer(self):
        if self.observation_file is not None and self.observation_buffer_rows:
            self.observation_file.write(
                self.observation_buffer[: self.observation_buffer_rows].tobytes()
            )
            self.observation_buffer_rows = 0

    def _close_observation_spool(self, *, incomplete=False):
        observation_file = getattr(self, "observation_file", None)
        if observation_file is not None:
            observation_file.close()
            self.observation_file = None
        self.observation_buffer = None
        self.observation_buffer_rows = 0
        observation_path = getattr(self, "observation_path", None)
        if observation_path is not None:
            try:
                os.unlink(observation_path)
            except FileNotFoundError:
                pass
            self.observation_path = None
        if incomplete:
            self.observation_complete = False

    def _record_observation(self, molecule_index, species_id, frames):
        if self.observation_file is None:
            return
        projected_rows = self.observation_rows + len(frames)
        if (
            projected_rows * self.observation_dtype.itemsize
            > _SPECIES_OBSERVATION_SPOOL_BYTES
        ):
            self._close_observation_spool(incomplete=True)
            return
        priority = self.molecule_count - molecule_index
        if len(frames) == 1:
            self.observation_buffer[self.observation_buffer_rows] = (
                int(frames[0]),
                species_id,
                priority,
            )
            self.observation_buffer_rows += 1
            if self.observation_buffer_rows == len(self.observation_buffer):
                self._flush_observation_buffer()
            self.observation_rows = projected_rows
            return
        source_start = 0
        while source_start < len(frames):
            available = len(self.observation_buffer) - self.observation_buffer_rows
            row_count = min(available, len(frames) - source_start)
            source_stop = source_start + row_count
            target_stop = self.observation_buffer_rows + row_count
            target = self.observation_buffer[self.observation_buffer_rows : target_stop]
            target["frame"] = frames[source_start:source_stop]
            target["species"] = species_id
            target["priority"] = priority
            self.observation_buffer_rows = target_stop
            source_start = source_stop
            if self.observation_buffer_rows == len(self.observation_buffer):
                self._flush_observation_buffer()
        self.observation_rows = projected_rows

    def _iter_observations(self):
        if not self.observation_complete or self.observation_file is None:
            raise RuntimeError("Species observation spool is incomplete")
        self._flush_observation_buffer()
        self.observation_file.flush()
        self.observation_file.seek(0)
        while True:
            rows = np.fromfile(
                self.observation_file,
                dtype=self.observation_dtype,
                count=_SPECIES_EVENT_CHUNK_ROWS,
            )
            if rows.size == 0:
                return
            yield rows

    def _create_temp_path(self):
        handle, self.path = tempfile.mkstemp(
            prefix="reacnetgenerator-species-",
            suffix=".mmap",
        )
        os.close(handle)

    def _allocate_dense(self):
        self.mode = "dense"
        self._create_temp_path()
        try:
            self.data = np.memmap(
                self.path,
                mode="w+",
                dtype=self.cell_dtype,
                shape=(self.species_count, self.frame_count),
            )
            self._update_peak_temporary_bytes()
        except BaseException:
            self.close()
            raise

    def _update_peak_temporary_bytes(self):
        final_bytes = os.path.getsize(self.path) if self.path is not None else 0
        observation_bytes = (
            self.observation_rows * self.observation_dtype.itemsize
            if self.observation_file is not None
            else 0
        )
        self.peak_temporary_bytes = max(
            self.peak_temporary_bytes,
            final_bytes + observation_bytes,
        )

    def _add_dense(self, molecule_index, species_id, frames):
        priority = self.molecule_count - molecule_index
        for chunk in self._chunks(frames):
            np.add.at(self.data["count"][species_id], chunk, 1)
            np.maximum.at(
                self.data["priority"][species_id],
                chunk,
                priority,
            )

    def observe(self, molecule_index, species_id, frames):
        """Count frame-block rows before choosing the smaller disk layout."""
        if self.mode not in (None, "dense"):
            raise RuntimeError("Cannot observe species rows after allocation")
        species_id = int(species_id)
        if species_id < 0 or species_id >= self.species_count:
            raise IndexError("Species ID is outside the name table")
        molecule_index = int(molecule_index)
        if molecule_index < 0 or molecule_index >= self.molecule_count:
            raise IndexError("Molecule index is outside the name table")
        frames = self._frame_array(frames)
        self._record_observation(molecule_index, species_id, frames)
        if self.mode == "dense":
            self._add_dense(molecule_index, species_id, frames)
            self.total_rows += len(frames)
            self.added_rows += len(frames)
            return
        if len(frames) == 1:
            block_id = int(frames[0]) // self.block_frame_count
            self.block_rows[block_id] += 1
            self.total_rows += 1
            return
        for chunk in self._chunks(frames):
            block_ids, counts = np.unique(
                chunk // self.block_frame_count,
                return_counts=True,
            )
            np.add.at(
                self.block_rows,
                block_ids.astype(np.intp, copy=False),
                counts.astype(np.uint64, copy=False),
            )
        self.total_rows += len(frames)

    def refine_sparse_blocks(self):
        """Shrink sparse frame partitions until stable sorting is memory-bounded."""
        if self.mode == "dense" or self.total_rows == 0:
            return False
        sparse_bytes = self.total_rows * self.event_dtype.itemsize
        if self.dense_bytes <= sparse_bytes:
            return False
        bytes_per_sorted_row = self._sparse_working_bytes_per_row
        row_limit = max(1, _SPECIES_COUNT_BLOCK_BYTES // bytes_per_sorted_row)
        while True:
            maximum_rows = int(self.block_rows.max(initial=0))
            if maximum_rows <= row_limit or self.block_frame_count == 1:
                return False
            refined_frame_count = max(
                1,
                self.block_frame_count * row_limit // maximum_rows,
            )
            if refined_frame_count >= self.block_frame_count:
                refined_frame_count = self.block_frame_count - 1
            self.block_frame_count = refined_frame_count
            self.block_count = (
                self.frame_count + self.block_frame_count - 1
            ) // self.block_frame_count
            self.block_rows = np.zeros(self.block_count, dtype=np.uint64)
            if (
                not self.observation_complete
                or self.observation_rows != self.total_rows
            ):
                self.total_rows = 0
                return True
            for rows in self._iter_observations():
                block_ids, counts = np.unique(
                    rows["frame"] // self.block_frame_count,
                    return_counts=True,
                )
                np.add.at(
                    self.block_rows,
                    block_ids.astype(np.intp, copy=False),
                    counts.astype(np.uint64, copy=False),
                )

    def allocate(self):
        """Choose dense counts or sparse events from their exact disk sizes."""
        if self.mode == "dense":
            return
        if self.mode is not None:
            raise RuntimeError("Species timeline store is already allocated")
        sparse_bytes = self.total_rows * self.event_dtype.itemsize
        if self.total_rows == 0 or self.frame_count == 0 or self.species_count == 0:
            self.mode = "empty"
            self._close_observation_spool()
            return
        if self.dense_bytes <= sparse_bytes:
            self._allocate_dense()
            self._populate_from_observations()
            return
        self.mode = "sparse"
        self._create_temp_path()
        try:
            self.block_offsets = np.empty(self.block_count + 1, dtype=np.uint64)
            self.block_offsets[0] = 0
            np.cumsum(self.block_rows, out=self.block_offsets[1:])
            self.block_cursors = self.block_offsets[:-1].copy()
            self.data = np.memmap(
                self.path,
                mode="w+",
                dtype=self.event_dtype,
                shape=(self.total_rows,),
            )
            self._update_peak_temporary_bytes()
            self._populate_from_observations()
        except BaseException:
            self.close()
            raise

    def _populate_from_observations(self):
        if (
            not self.observation_complete
            or self.observation_file is None
            or self.observation_rows != self.total_rows
        ):
            return
        for rows in self._iter_observations():
            frame_ids = rows["frame"].astype(np.intp)
            species_ids = rows["species"].astype(np.intp)
            if self.mode == "dense":
                np.add.at(self.data["count"], (species_ids, frame_ids), 1)
                np.maximum.at(
                    self.data["priority"],
                    (species_ids, frame_ids),
                    rows["priority"],
                )
            else:
                self._populate_sparse_rows(rows, frame_ids)
            self.added_rows += len(rows)
        self.observation_reused = True
        self._close_observation_spool()

    def _populate_sparse_rows(self, rows, frame_ids):
        """Scatter one observation chunk by block without repeated full scans."""
        row_count = len(rows)
        if row_count == 0:
            return
        block_ids = frame_ids // self.block_frame_count
        if row_count == 1 or np.all(block_ids[1:] >= block_ids[:-1]):
            order = None
            sorted_block_ids = block_ids
        else:
            order = np.argsort(block_ids, kind="stable")
            sorted_block_ids = block_ids[order]
        boundaries = np.flatnonzero(sorted_block_ids[1:] != sorted_block_ids[:-1]) + 1
        group_starts = np.concatenate((np.zeros(1, dtype=np.intp), boundaries))
        group_stops = np.concatenate(
            (boundaries, np.asarray([row_count], dtype=np.intp))
        )
        group_counts = group_stops - group_starts
        unique_blocks = sorted_block_ids[group_starts]
        cursor_starts = self.block_cursors[unique_blocks].astype(
            np.intp,
            copy=False,
        )
        target_positions = np.arange(row_count, dtype=np.intp)
        target_positions += np.repeat(
            cursor_starts - group_starts,
            group_counts,
        )
        if order is None:
            self.data["frame"][target_positions] = rows["frame"]
            self.data["species"][target_positions] = rows["species"]
        else:
            self.data["frame"][target_positions] = rows["frame"][order]
            self.data["species"][target_positions] = rows["species"][order]
        self.block_cursors[unique_blocks] += group_counts.astype(
            np.uint64,
            copy=False,
        )

    def add(self, molecule_index, species_id, frames):
        """Add one molecule timeline during the second sequential file pass."""
        if self.mode is None:
            raise RuntimeError("Species timeline store has not been allocated")
        species_id = int(species_id)
        if species_id < 0 or species_id >= self.species_count:
            raise IndexError("Species ID is outside the name table")
        molecule_index = int(molecule_index)
        if molecule_index < 0 or molecule_index >= self.molecule_count:
            raise IndexError("Molecule index is outside the name table")
        frames = self._frame_array(frames)
        if self.mode == "empty":
            if frames.size:
                raise RuntimeError("Observed and written species row counts differ")
            return
        if self.mode == "dense":
            self._add_dense(molecule_index, species_id, frames)
        else:
            for chunk in self._chunks(frames):
                block_ids = chunk // self.block_frame_count
                boundaries = np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1
                starts = np.concatenate((np.zeros(1, dtype=np.intp), boundaries))
                stops = np.concatenate(
                    (boundaries, np.asarray([len(chunk)], dtype=np.intp))
                )
                for start, stop in zip(starts, stops):
                    block_id = int(block_ids[start])
                    target_start = int(self.block_cursors[block_id])
                    target_stop = target_start + int(stop - start)
                    target = self.data[target_start:target_stop]
                    target["frame"] = chunk[start:stop]
                    target["species"] = species_id
                    self.block_cursors[block_id] = target_stop
        self.added_rows += len(frames)

    @property
    def requires_second_pass(self):
        return self.added_rows != self.total_rows

    def _check_complete(self):
        if self.added_rows != self.total_rows:
            raise RuntimeError("Observed and written species row counts differ")
        if self.mode == "sparse" and not np.array_equal(
            self.block_cursors,
            self.block_offsets[1:],
        ):
            raise RuntimeError("Species event block offsets are incomplete")

    def iter_frame_counts(self):
        """Yield ordered species counts one frame at a time with bounded memory."""
        self._check_complete()
        for block_index in range(self.block_count):
            frame_start = block_index * self.block_frame_count
            frame_stop = min(
                frame_start + self.block_frame_count,
                self.frame_count,
            )
            if self.mode == "dense":
                count_block = np.ascontiguousarray(
                    self.data[:, frame_start:frame_stop].T
                )
                for frame_offset, row in enumerate(count_block):
                    active_species = np.flatnonzero(row["count"])
                    active_species = active_species[
                        np.argsort(row["priority"][active_species])[::-1]
                    ]
                    yield (
                        frame_start + frame_offset,
                        (
                            (int(species_id), int(row["count"][species_id]))
                            for species_id in active_species
                        ),
                    )
                continue
            if self.mode == "empty":
                for frame in range(frame_start, frame_stop):
                    yield frame, ()
                continue
            row_start = int(self.block_offsets[block_index])
            row_stop = int(self.block_offsets[block_index + 1])
            rows = self.data[row_start:row_stop]
            order = np.argsort(rows["frame"], kind="stable")
            sorted_frames = rows["frame"][order]
            sorted_species = rows["species"][order]
            event_start = 0
            for frame in range(frame_start, frame_stop):
                if (
                    event_start >= len(sorted_frames)
                    or int(sorted_frames[event_start]) != frame
                ):
                    yield frame, ()
                    continue
                event_stop = int(np.searchsorted(sorted_frames, frame, side="right"))
                yield (
                    frame,
                    self._iter_sparse_counts(sorted_species[event_start:event_stop]),
                )
                event_start = event_stop

    @staticmethod
    def _iter_sparse_counts(species_ids):
        """Aggregate one frame without Python objects per active species."""
        if len(species_ids) == 1:
            yield int(species_ids[0]), 1
            return
        unique_species, first_indices, counts = np.unique(
            species_ids,
            return_index=True,
            return_counts=True,
        )
        for index in np.argsort(first_indices):
            yield int(unique_species[index]), int(counts[index])

    @property
    def _sparse_working_bytes_per_row(self):
        """Conservatively estimate stable-sort and unique scratch per event."""
        return (
            10 * np.dtype(np.intp).itemsize
            + 4 * self.event_dtype.itemsize
            + np.dtype(np.bool_).itemsize
        )

    @property
    def temporary_bytes(self):
        if self.path is None:
            return 0
        return os.path.getsize(self.path)

    @property
    def maximum_count_block_bytes(self):
        if self.mode == "sparse":
            return int(self.block_rows.max(initial=0)) * (
                self._sparse_working_bytes_per_row
            )
        return min(self.frame_count, self.block_frame_count) * (
            self.species_count * self.cell_dtype.itemsize
        )

    def close(self):
        self._close_observation_spool()
        data = getattr(self, "data", None)
        if data is not None:
            data.flush()
            mmap = getattr(data, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self.data = None
        path = getattr(self, "path", None)
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            self.path = None


class _GenerateMatrix(SharedRNGData):
    tablefilename: str
    speciesfilename: str
    reactionfilename: str
    moleculetemp2filename: str
    n_searchspecies: int
    needprintspecies: bool
    allmoleculeroute: np.ndarray | Counter
    speciescenter: str
    matrix_size: int
    mname: np.ndarray | _MoleculeNameTable
    timestep: np.ndarray
    splitmoleculeroute: list[np.ndarray | Counter]

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            [
                "tablefilename",
                "speciesfilename",
                "reactionfilename",
                "moleculetemp2filename",
                "n_searchspecies",
                "needprintspecies",
                "allmoleculeroute",
                "speciescenter",
                "mname",
                "timestep",
                "splitmoleculeroute",
                "matrix_size",
            ],
            [],
        )

    def generate(self):
        """Generate a reaction matrix and print species.

        A reaction matrix can be generated as
            R=[a_ij ], i=1,2,…,100;j=1,2,…,100
        where aij is the number of reactions from species si to sj.
        """
        self._printtable(self._getallroute(self.allmoleculeroute))
        if self.splitmoleculeroute is not None:
            for i, smr in enumerate(self.splitmoleculeroute):
                self._printtable(self._getallroute(smr), timeaxis=i)
        if self.needprintspecies:
            self._printspecies()

    def _getallroute(self, allmoleculeroute):
        if isinstance(allmoleculeroute, Counter):
            assert isinstance(self.mname, _MoleculeNameTable)
            return (
                (
                    [
                        str(self.mname.names[int(left)]),
                        str(self.mname.names[int(right)]),
                    ],
                    int(count),
                )
                for (left, right), count in allmoleculeroute.items()
            )
        if isinstance(self.mname, _MoleculeNameTable):
            routes = np.asarray(allmoleculeroute).reshape((-1, 2))
            name_ids = self.mname.ids[routes - 1]
            name_ids = name_ids[name_ids[:, 0] != name_ids[:, 1]]
            if name_ids.size == 0:
                return []
            pairs, counts = np.unique(name_ids, return_counts=True, axis=0)
            return (
                (
                    [
                        str(self.mname.names[int(left)]),
                        str(self.mname.names[int(right)]),
                    ],
                    int(count),
                )
                for (left, right), count in zip(pairs, counts)
            )
        names = self.mname[allmoleculeroute - 1]
        names = names[names[:, 0] != names[:, 1]]
        if names.size > 0:
            equations = np.unique(names, return_counts=True, axis=0)
            return zip(equations[0].tolist(), equations[1].tolist())
        return []

    def _printtable(self, allroute, timeaxis=None):
        maxsize = self.matrix_size
        species = []
        sortedreactions = sorted(allroute, key=operator.itemgetter(1, 0), reverse=True)
        # added on Nov 17, 2018
        if self.speciescenter:
            newreactions = []
            species = [self.speciescenter]
            newspecies = [self.speciescenter]
            while len(species) < maxsize and newspecies:
                newnewspecies = []
                for newspec in newspecies:
                    searchedspecies = self._searchspecies(
                        newspec, sortedreactions, species
                    )
                    for searchedspec, searchedreaction in searchedspecies:
                        if len(species) < maxsize:
                            newnewspecies.append(searchedspec)
                            species.append(searchedspec)
                            newreactions.append(searchedreaction)
                newspecies = newnewspecies
            for reac in sortedreactions:
                if reac not in newreactions:
                    newreactions.append(reac)
            sortedreactions = newreactions

        table = np.zeros((maxsize, maxsize), dtype=int)
        reactionnumber = np.zeros((2), dtype=int)
        with open(
            (
                self.reactionfilename
                if timeaxis is None
                else f"{self.reactionfilename}.{timeaxis}"
            ),
            "w",
        ) as f:
            for reaction, n_reaction in sortedreactions:
                f.write(f"{n_reaction} {'->'.join(reaction)}\n")
                for i, spec in enumerate(reaction):
                    if spec in species:
                        number = species.index(spec)
                    elif len(species) < maxsize:
                        species.append(spec)
                        number = species.index(spec)
                    else:
                        number = -1
                    reactionnumber[i] = number
                if all(reactionnumber >= 0):
                    table[tuple(reactionnumber)] = n_reaction

        species_idx = pd.Index(species)
        df = pd.DataFrame(
            table[: len(species), : len(species)],
            index=species_idx,
            columns=species_idx,
        )
        df.to_csv(
            (
                self.tablefilename
                if timeaxis is None
                else f"{self.tablefilename}.{timeaxis}"
            ),
            sep=" ",
        )

    def _searchspecies(self, originspec, sortedreactions, species):
        searchedspecies = []
        for reaction, n_reaction in sortedreactions:
            ii = 1
            if originspec == reaction[1 - ii]:
                if reaction[ii] not in species:
                    searchedspecies.append((reaction[ii], (reaction, n_reaction)))
            if len(searchedspecies) >= self.n_searchspecies:
                break
        return searchedspecies

    def _printspecies(self):
        names = (
            self.mname
            if isinstance(self.mname, _MoleculeNameTable)
            else _MoleculeNameTable.from_names(self.mname)
        )
        with _SpeciesTimelineStore(
            len(self.timestep),
            len(names.names),
            len(names),
        ) as store:
            for molecule_index, species_id, frames in self._iter_species_frames(names):
                store.observe(molecule_index, species_id, frames)
            while store.refine_sparse_blocks():
                for molecule_index, species_id, frames in self._iter_species_frames(
                    names
                ):
                    store.observe(molecule_index, species_id, frames)
            store.allocate()
            if store.requires_second_pass:
                for molecule_index, species_id, frames in self._iter_species_frames(
                    names
                ):
                    store.add(molecule_index, species_id, frames)
            logger.info(
                "Species timeline: %d molecule-frame rows, %d species, "
                "%s temporary layout, observation spool reused=%s, "
                "%.3f MiB final / %.3f MiB peak temporary, "
                "%.3f MiB maximum aggregation block",
                store.total_rows,
                len(names.names),
                store.mode,
                store.observation_reused,
                store.temporary_bytes / (1024 * 1024),
                store.peak_temporary_bytes / (1024 * 1024),
                store.maximum_count_block_bytes / (1024 * 1024),
            )
            with WriteBuffer(open(self.speciesfilename, "w")) as output:
                for frame, counts in store.iter_frame_counts():
                    output.append(f"Timestep {self.timestep[frame]}:")
                    output.extend(
                        f" {names.names[species_id]} {count}"
                        for species_id, count in counts
                    )
                    output.append("\n")

    def _iter_species_frames(self, names):
        with open(self.moleculetemp2filename, "rb") as molecule_file:
            frame_blocks = _iter_compressed_record_fields(molecule_file, (3,))
            for molecule_index, (species_id, (frame_block,)) in enumerate(
                zip(names.ids, frame_blocks)
            ):
                yield (
                    molecule_index,
                    int(species_id),
                    np.asarray(bytestolist(frame_block)).reshape((-1,)),
                )
