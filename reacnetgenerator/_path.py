# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Collect paths.

To produce a reaction network, every molecule (species) should be treated as a
node in the network. Therefore, all detected species are indexed by canonical
SMILES to guarantee its uniqueness. Isomers are also identified according to
SMILES codes._[1] The VF2 algorithm can be also used to identify isomers, which is
an option in ReacNetGenerator._[2] After filtering out noise, the reaction path of atoms
and the number of intermolecular reactions can be calculated.

References
----------
.. [1] Landrum, G. RDKit: Open-Source Cheminformatics Software 2016.
.. [2] Cordella, L. P.; Foggia, P.; Sansone, C.; Vento, M. A (Sub)Graph
   Isomorphism Algorith for Matching Large Graphs. IEEE Trans. Pattern Analysis
   and Machine Intelligence 2004, 26, 1367-1372.
"""

import itertools
import os
import re
import tempfile
import time
from abc import ABCMeta, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from multiprocessing import get_start_method
from typing import NamedTuple

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from ._hmmfilter import _SmilesWorkMetrics
from ._logging import logger
from ._moleculenames import (
    _MoleculeNameBuilder,
    _MoleculeNameTable,
    _unsigned_dtype_for_maximum,
)
from ._packedbool import _PackedBoolMatrix
from ._reaction import ReactionsFinder
from ._timedoutput import TimedOutputStore
from .utils import (
    SharedRNGData,
    WriteBuffer,
    _frame_indices_fit_signal_memory,
    _iter_compressed_record_fields,
    bytestolist,
    get_timestep_value,
    read_compressed_block,
    run_mp,
)

_MATRIX_WRITE_CELLS = 1024 * 1024
_MATRIX_WRITE_ATOMS = 65536
_MATRIX_FRAME_SCAN_ROWS = 1024 * 1024
_MATRIX_DENSE_PREFIX_ROWS = 64
_RANGE_OUTPUT_BLOCK_ROWS = 4096
_RANGE_SCAN_ROWS = 65536
_RANGE_LINEAR_SCAN_ROWS = 64
_SMILES_AVERAGE_STRUCTURE_BYTES_PER_WORKER = 2048
_SMILES_FORK_BATCH_AVERAGE_STRUCTURE_BYTES = 512
_SMILES_FORK_CHUNKSIZE = 64
_SMILES_FORK_LIGHT_STRUCTURE_BYTES_PER_WORKER = 24 * 1024 * 1024
_SMILES_FORK_MAX_LIGHT_WORKERS = 4
_SMILES_FORK_REUSE_MAX_STRUCTURE_BYTES = 64 * 1024
_SMILES_NAME_CACHE_BYTES = 1024 * 1024
_SMILES_NAME_CACHE_ENTRY_OVERHEAD = 512
_SMILES_NAME_CACHE_ATOM_BYTES = np.dtype(np.int64).itemsize
_SMILES_NAME_CACHE_BOND_BYTES = 3 * np.dtype(np.int64).itemsize
_SMILES_TOTAL_STRUCTURE_BYTES_PER_WORKER = 3 * 1024 * 1024
_ROUTE_COMPACTION_BLOCK_ROWS = 65536
_ROUTE_COMPACTION_MAX_UNIQUE_PAIRS = 65536
_ROUTE_COMPACTION_MIN_ROWS = 256
_ROUTE_COMPACTION_SAMPLE_ROWS = 1024
_ROUTE_CHANGE_SCAN_ROWS = 65536
_ROUTE_ATOMS_PER_WORKER = 64
_ROUTE_SCAN_VALUES_PER_WORKER = 65536
_ROUTE_WRITE_BUFFER_BYTES = 8 * 1024 * 1024
_ROUTE_ATOMEACH = None
_ROUTE_ATOMTYPE = None
_ROUTE_ATOMNAME = None
_ROUTE_SELECTATOMS = None
_ROUTE_MNAME = None
_ROUTE_FRAME_RANGE = None
_ROUTE_ACTIVE_TRANSITIONS = None
_SMILES_WORKER = None

_iter_molecule_fields = _iter_compressed_record_fields


class _SmilesResultRecord(NamedTuple):
    """Parent-side SMILES result with explicit compressed/decoded states."""

    name: str | None
    structure_record: tuple[bytes, ...] | None
    frame_block: bytes | None
    atoms: np.ndarray | None
    bonds: list | None

    @classmethod
    def from_compressed(cls, name, record):
        """Keep a parent record for structure decoding and optional timeline."""
        if len(record) not in (3, 4):
            raise ValueError("SMILES parent record must contain 3 or 4 fields")
        return cls(
            name=name,
            structure_record=record,
            frame_block=record[3] if len(record) == 4 else None,
            atoms=None,
            bonds=None,
        )

    @classmethod
    def from_decoded(cls, name, atoms, bonds, frame_block=None):
        """Keep decoded structure fields and an optional timeline payload."""
        if atoms is None or bonds is None:
            raise ValueError("Decoded SMILES results require atoms and bonds")
        return cls(
            name=name,
            structure_record=None,
            frame_block=frame_block,
            atoms=atoms,
            bonds=bonds,
        )


def _write_atom_frame_selection(store, atoms, frames, molecule_id):
    """Write one bounded molecule selection and preserve overlap conflicts."""
    matrix_index = (
        (atoms, frames) if isinstance(frames, slice) else np.ix_(atoms, frames)
    )
    selected = store.atomeach[matrix_index]
    overlap = selected != 0
    if np.any(overlap):
        store.conflict.mark(atoms, frames, overlap)
    store.atomeach[matrix_index] = molecule_id


def _write_indexed_atom_frames(store, atoms, frames, molecule_id):
    """Write bounded sparse frame-index selections for one molecule."""
    for atom_start in range(0, len(atoms), _MATRIX_WRITE_ATOMS):
        atom_batch = atoms[atom_start : atom_start + _MATRIX_WRITE_ATOMS]
        frames_per_batch = max(
            1,
            _MATRIX_WRITE_CELLS // max(1, len(atom_batch)),
        )
        for frame_start in range(0, len(frames), frames_per_batch):
            frame_batch = frames[frame_start : frame_start + frames_per_batch]
            _write_atom_frame_selection(
                store,
                atom_batch,
                frame_batch,
                molecule_id,
            )


def _write_dense_atom_frame_span(store, atoms, frame_start, frame_stop, molecule_id):
    """Write one bounded continuous frame span without materializing indices."""
    for atom_start in range(0, len(atoms), _MATRIX_WRITE_ATOMS):
        atom_batch = atoms[atom_start : atom_start + _MATRIX_WRITE_ATOMS]
        frames_per_batch = max(
            1,
            _MATRIX_WRITE_CELLS // max(1, len(atom_batch)),
        )
        for batch_start in range(frame_start, frame_stop, frames_per_batch):
            batch_stop = min(batch_start + frames_per_batch, frame_stop)
            _write_atom_frame_selection(
                store,
                atom_batch,
                slice(batch_start, batch_stop),
                molecule_id,
            )


def _is_dense_signal_block(signal_block):
    """Confirm density after rejecting common sparse blocks from a prefix."""
    if len(signal_block) == 0:
        return False
    prefix = signal_block[:_MATRIX_DENSE_PREFIX_ROWS]
    if not np.all(prefix):
        return False
    if len(signal_block) <= len(prefix):
        return True
    if not signal_block[-1]:
        return False
    return bool(np.all(signal_block))


def _use_direct_no_hmm_frames(frame_block, frame_count):
    """Prefer stored indices only when their decoded size is signal-bounded."""
    return _frame_indices_fit_signal_memory(frame_block, frame_count)


def _iter_atom_frame_records(signal_blocks, molecule_records, run_hmm, frame_count):
    """Pair only signal-backed records; direct no-HMM records consume none."""
    signals = iter(signal_blocks)
    for record in molecule_records:
        direct_frames = not run_hmm and _use_direct_no_hmm_frames(
            record[1],
            frame_count,
        )
        if direct_frames:
            yield None, record
            continue
        try:
            signal_block = next(signals)
        except StopIteration as error:
            raise EOFError("Missing atom-frame signal block") from error
        yield signal_block, record


def _initialize_smiles_worker(atomname, atomtype):
    """Build one minimal SMILES converter per worker process."""
    global _SMILES_WORKER
    worker = object.__new__(_CollectSMILESPaths)
    worker.atomname = np.asarray(atomname)
    worker.atomtype = np.asarray(atomtype)
    worker.atomnames = worker.atomname[worker.atomtype]
    _SMILES_WORKER = worker


def _get_smiles_name(record):
    """Multiprocessing entry point returning only one compact species name."""
    assert _SMILES_WORKER is not None
    return _SMILES_WORKER._calmoleculeSMILESname(record)


def _get_smiles_name_and_structure(record):
    """Return a name plus decoded structure for the cheap batched fork path."""
    assert _SMILES_WORKER is not None
    atoms, bonds = _SMILES_WORKER._getatomsandbonds(record)
    name = _SMILES_WORKER._calmoleculeSMILESname_from_decoded(atoms, bonds)
    return name, atoms, bonds


def _smiles_worker_count(
    requested_nproc,
    molecule_count,
    structure_bytes,
    start_method=None,
):
    """Select workers from per-record complexity and total amortized work."""
    requested_nproc = int(requested_nproc)
    molecule_count = max(0, int(molecule_count))
    maximum_workers = min(requested_nproc, max(1, molecule_count))
    if structure_bytes is None:
        return maximum_workers
    structure_bytes = max(0, int(structure_bytes))
    average_bytes = (
        (structure_bytes + molecule_count - 1) // molecule_count
        if molecule_count
        else 0
    )
    complexity_workers = max(
        1,
        (average_bytes + _SMILES_AVERAGE_STRUCTURE_BYTES_PER_WORKER - 1)
        // _SMILES_AVERAGE_STRUCTURE_BYTES_PER_WORKER,
    )
    amortized_workers = max(
        1,
        structure_bytes // _SMILES_TOTAL_STRUCTURE_BYTES_PER_WORKER,
    )
    if start_method is None:
        start_method = get_start_method()
    if start_method != "fork":
        return min(maximum_workers, complexity_workers, amortized_workers)
    light_workers = min(
        _SMILES_FORK_MAX_LIGHT_WORKERS,
        max(
            1,
            (structure_bytes + _SMILES_FORK_LIGHT_STRUCTURE_BYTES_PER_WORKER - 1)
            // _SMILES_FORK_LIGHT_STRUCTURE_BYTES_PER_WORKER,
        ),
    )
    return min(
        maximum_workers,
        amortized_workers,
        max(complexity_workers, light_workers),
    )


def _smiles_pool_limits(
    worker_count,
    molecule_count,
    structure_bytes,
    start_method=None,
):
    """Batch only cheap fork records and bound the submitted record window."""
    worker_count = max(1, int(worker_count))
    molecule_count = max(0, int(molecule_count))
    if start_method is None:
        start_method = get_start_method()
    chunksize = 1
    if start_method == "fork" and structure_bytes is not None and molecule_count:
        average_bytes = (
            max(0, int(structure_bytes)) + molecule_count - 1
        ) // molecule_count
        if average_bytes <= _SMILES_FORK_BATCH_AVERAGE_STRUCTURE_BYTES:
            chunksize = min(
                _SMILES_FORK_CHUNKSIZE,
                max(1, molecule_count // worker_count),
            )
    max_inflight = max(chunksize, 2 * worker_count * chunksize)
    return chunksize, max_inflight


def _should_reuse_smiles_worker_structures(
    worker_count,
    chunksize,
    start_method,
    maximum_structure_bytes,
):
    """Reuse decoded records only on the benchmarked cheap batched fork path."""
    return (
        int(worker_count) > 1
        and start_method == "fork"
        and int(chunksize) > 1
        and maximum_structure_bytes is not None
        and int(maximum_structure_bytes) <= _SMILES_FORK_REUSE_MAX_STRUCTURE_BYTES
    )


class _AtomFrameStore:
    """Disk-backed atom-by-frame matrices used only during PATH analysis."""

    def __init__(self, shape, molecule_dtype):
        self.shape = tuple(int(value) for value in shape)
        self.molecule_dtype = np.dtype(molecule_dtype)
        atom_handle, self.atomeach_path = tempfile.mkstemp(
            prefix="reacnetgenerator-atomeach-", suffix=".mmap"
        )
        conflict_handle, self.conflict_path = tempfile.mkstemp(
            prefix="reacnetgenerator-conflict-", suffix=".mmap"
        )
        os.close(atom_handle)
        os.close(conflict_handle)
        self.atomeach = np.memmap(
            self.atomeach_path,
            mode="w+",
            dtype=self.molecule_dtype,
            shape=self.shape,
        )
        self.conflict = _PackedBoolMatrix(
            self.conflict_path,
            self.shape,
            mode="w+",
        )
        self.active_transition_path = None
        self.active_transitions = np.zeros(0, dtype=np.uint8)
        transition_count = max(0, self.shape[1] - 1)
        if transition_count:
            # Keep one byte per transition so workers only perform idempotent
            # byte stores; packed bits would require a racy shared read/modify/write.
            active_handle, self.active_transition_path = tempfile.mkstemp(
                prefix="reacnetgenerator-active-transitions-",
                suffix=".mmap",
            )
            os.close(active_handle)
            self.active_transitions = np.memmap(
                self.active_transition_path,
                mode="w+",
                dtype=np.uint8,
                shape=(transition_count,),
            )
            self.active_transitions[:] = 0
        self.mname_path = None
        self.mname_values_path = None
        self.atomtype_path = None

    def save_molecule_names(self, values):
        """Save compact molecule IDs and unique names for read-only workers."""
        if not isinstance(values, _MoleculeNameTable):
            values = _MoleculeNameTable.from_names(values)
        handle, self.mname_path = tempfile.mkstemp(
            prefix="reacnetgenerator-mname-ids-", suffix=".npy"
        )
        os.close(handle)
        handle, self.mname_values_path = tempfile.mkstemp(
            prefix="reacnetgenerator-mname-values-", suffix=".npy"
        )
        os.close(handle)
        np.save(self.mname_path, values.ids, allow_pickle=False)
        np.save(self.mname_values_path, values.names, allow_pickle=False)

    def save_atom_types(self, values):
        """Save per-atom types once for route workers to map read-only."""
        handle, self.atomtype_path = tempfile.mkstemp(
            prefix="reacnetgenerator-atomtype-", suffix=".npy"
        )
        os.close(handle)
        np.save(self.atomtype_path, np.asarray(values), allow_pickle=False)

    def flush(self):
        self.atomeach.flush()
        self.conflict.flush()
        if self.active_transition_path is not None:
            self.active_transitions.flush()

    def close(self):
        """Close mappings and remove their temporary backing files."""
        value = getattr(self, "atomeach", None)
        if value is not None:
            value.flush()
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self.atomeach = None
        conflict = getattr(self, "conflict", None)
        if conflict is not None:
            conflict.close()
            self.conflict = None
        active_transitions = getattr(self, "active_transitions", None)
        if self.active_transition_path is not None and active_transitions is not None:
            active_transitions.flush()
            mmap = getattr(active_transitions, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.active_transitions = None
        for path in (
            self.atomeach_path,
            self.conflict_path,
            self.active_transition_path,
            self.mname_path,
            self.mname_values_path,
            self.atomtype_path,
        ):
            if path is None:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _initialize_route_worker(
    atomeach_path,
    shape,
    dtype_string,
    mname_path,
    mname_values_path,
    atomtype_path,
    atomname,
    selectatoms,
    frame_range,
    active_transition_path=None,
    active_transition_count=0,
):
    """Attach one route worker to read-only shared mappings."""
    global _ROUTE_ATOMEACH
    global _ROUTE_ATOMTYPE
    global _ROUTE_ATOMNAME
    global _ROUTE_SELECTATOMS
    global _ROUTE_MNAME
    global _ROUTE_FRAME_RANGE
    global _ROUTE_ACTIVE_TRANSITIONS
    _ROUTE_ATOMEACH = np.memmap(
        atomeach_path,
        mode="r",
        dtype=np.dtype(dtype_string),
        shape=tuple(shape),
    )
    _ROUTE_ATOMTYPE = np.load(atomtype_path, mmap_mode="r", allow_pickle=False)
    _ROUTE_ATOMNAME = np.asarray(atomname)
    _ROUTE_SELECTATOMS = set(selectatoms)
    _ROUTE_MNAME = _MoleculeNameTable(
        np.load(mname_path, mmap_mode="r", allow_pickle=False),
        np.load(mname_values_path, mmap_mode="r", allow_pickle=False),
        validate=False,
    )
    _ROUTE_FRAME_RANGE = frame_range
    _ROUTE_ACTIVE_TRANSITIONS = None
    if active_transition_path is not None and active_transition_count:
        _ROUTE_ACTIVE_TRANSITIONS = np.memmap(
            active_transition_path,
            mode="r+",
            dtype=np.uint8,
            shape=(int(active_transition_count),),
        )


def _count_timeline_changes(timeline, active_transitions=None):
    """Count adjacent changes with bounded temporary boolean blocks."""
    if active_transitions is not None and len(active_transitions) != max(
        0, len(timeline) - 1
    ):
        raise ValueError("Active transition index length does not match timeline")
    change_count = 0
    for change_start in range(1, len(timeline), _ROUTE_CHANGE_SCAN_ROWS):
        change_stop = min(change_start + _ROUTE_CHANGE_SCAN_ROWS, len(timeline))
        change_mask = np.empty(change_stop - change_start, dtype=np.bool_)
        np.not_equal(
            timeline[change_start:change_stop],
            timeline[change_start - 1 : change_stop - 1],
            out=change_mask,
        )
        block_change_count = int(np.count_nonzero(change_mask))
        change_count += block_change_count
        if active_transitions is not None:
            active_slice = active_transitions[change_start - 1 : change_stop - 1]
            if block_change_count == len(change_mask):
                active_slice[:] = 1
            elif block_change_count:
                active_slice[change_mask] = 1
    return change_count


def _calculate_atom_route_details(
    atom_index,
    timeline,
    atomtype,
    atomname,
    selectatoms,
    molecule_names,
    active_transitions=None,
):
    """Calculate one atom route and its exact transition-change count."""
    if active_transitions is not None and len(active_transitions) != max(
        0, len(timeline) - 1
    ):
        raise ValueError("Active transition index length does not match timeline")
    modified_atom_events = None
    if np.count_nonzero(timeline) != timeline.size:
        modified_atom_events = _count_timeline_changes(
            timeline,
            active_transitions,
        )
        timeline = timeline[timeline != 0]
    if timeline.size:
        change_mask = np.empty(timeline.size, dtype=np.bool_)
        change_mask[0] = True
        np.not_equal(timeline[1:], timeline[:-1], out=change_mask[1:])
        change_time = np.flatnonzero(change_mask)
        route = timeline[change_time]
    else:
        change_time = np.zeros(0, dtype=int)
        route = np.zeros(0, dtype=int)
    atom_type = int(atomtype[atom_index])
    atom_name = str(atomname[atom_type])
    molecule_route = (
        np.column_stack((route[:-1], route[1:]))
        if atom_name in selectatoms
        else np.zeros((0, 2), dtype=int)
    )
    names = molecule_names[route - 1]
    route_string = f"Atom {atom_index + 1} {atom_name}: " + " -> ".join(
        f"{frame} {name}" for frame, name in zip(change_time, names)
    )
    if modified_atom_events is None:
        modified_atom_events = max(0, len(route) - 1)
        if active_transitions is not None and modified_atom_events:
            if modified_atom_events == len(active_transitions):
                active_transitions[:] = 1
            else:
                active_transitions[change_time[1:] - 1] = 1
    return molecule_route, route_string, modified_atom_events


def _calculate_atom_route(
    atom_index,
    timeline,
    atomtype,
    atomname,
    selectatoms,
    molecule_names,
):
    """Calculate one atom route without owning the full atom-frame matrix."""
    molecule_route, route_string, _ = _calculate_atom_route_details(
        atom_index,
        timeline,
        atomtype,
        atomname,
        selectatoms,
        molecule_names,
    )
    return molecule_route, route_string


def _get_atom_route_by_index(atom_index):
    """Multiprocessing entry point receiving only an integer atom index."""
    assert _ROUTE_ATOMEACH is not None
    assert _ROUTE_FRAME_RANGE is not None
    start, stop = _ROUTE_FRAME_RANGE
    return _get_atom_route_result(
        int(atom_index),
        _ROUTE_ATOMEACH[int(atom_index), start:stop],
        _ROUTE_ATOMTYPE,
        _ROUTE_ATOMNAME,
        _ROUTE_SELECTATOMS,
        _ROUTE_MNAME,
        _ROUTE_ACTIVE_TRANSITIONS,
    )


def _compact_route_pairs(molecule_routes):
    """Aggregate repeated molecule pairs before returning a worker result."""
    route_count = len(molecule_routes)
    if route_count < _ROUTE_COMPACTION_MIN_ROWS:
        return molecule_routes, None
    sample_step = max(1, route_count // _ROUTE_COMPACTION_SAMPLE_ROWS)
    sample = molecule_routes[::sample_step][:_ROUTE_COMPACTION_SAMPLE_ROWS]
    if len(np.unique(sample, axis=0)) == len(sample):
        return molecule_routes, None
    pair_counts = Counter()
    for block_start in range(0, route_count, _ROUTE_COMPACTION_BLOCK_ROWS):
        block_pairs, block_counts = np.unique(
            molecule_routes[block_start : block_start + _ROUTE_COMPACTION_BLOCK_ROWS],
            axis=0,
            return_counts=True,
        )
        for pair, count in zip(block_pairs, block_counts):
            pair_counts[(int(pair[0]), int(pair[1]))] += int(count)
            if len(pair_counts) > _ROUTE_COMPACTION_MAX_UNIQUE_PAIRS:
                return molecule_routes, None
    ordered_pairs = sorted(pair_counts)
    route_pairs = np.asarray(ordered_pairs, dtype=molecule_routes.dtype).reshape(
        (-1, 2)
    )
    route_counts = np.fromiter(
        (pair_counts[pair] for pair in ordered_pairs),
        dtype=np.uint64,
        count=len(ordered_pairs),
    )
    return route_pairs, route_counts


def _get_atom_route_result(
    atom_index,
    timeline,
    atomtype,
    atomname,
    selectatoms,
    molecule_names,
    active_transitions=None,
):
    """Return one route result in the shared serial/worker representation."""
    molecule_routes, route_string, modified_atom_events = _calculate_atom_route_details(
        atom_index,
        timeline,
        atomtype,
        atomname,
        selectatoms,
        molecule_names,
        active_transitions,
    )
    route_pairs, route_counts = _compact_route_pairs(molecule_routes)
    return (
        route_pairs,
        route_counts,
        route_string.encode("utf-8"),
        modified_atom_events,
    )


def _route_worker_count(requested_nproc, atom_count, frame_count):
    """Keep enough atom tasks and atom-frame scan work behind each worker."""
    atom_count = max(0, int(atom_count))
    scan_values = atom_count * max(0, int(frame_count))
    atom_workers = max(
        1,
        (atom_count + _ROUTE_ATOMS_PER_WORKER - 1) // _ROUTE_ATOMS_PER_WORKER,
    )
    return min(
        int(requested_nproc),
        atom_workers,
        max(1, scan_values // _ROUTE_SCAN_VALUES_PER_WORKER),
    )


class _CollectPaths(SharedRNGData, metaclass=ABCMeta):
    runHMM: bool
    N: int
    step: int
    atomname: np.ndarray
    originfilename: str
    hmmfilename: str
    moleculefilename: str
    timedoutputfilename: str
    timedoutputcachemib: int
    moleculetemp2filename: str
    atomroutefilename: str
    nproc: int
    hmmit: int
    atomtype: np.ndarray
    selectatoms: list
    split: int
    miso: int
    timestep: dict
    framesource: dict
    inputfilename: list
    stepinterval: int
    printmoleculetime: bool
    moleculeframes: list
    moleculetimesteps: list
    printreactionevent: bool
    smilesworkmetrics: _SmilesWorkMetrics | None
    mname: _MoleculeNameTable

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            [
                "runHMM",
                "N",
                "step",
                "atomname",
                "originfilename",
                "hmmfilename",
                "moleculefilename",
                "timedoutputfilename",
                "timedoutputcachemib",
                "moleculetemp2filename",
                "atomroutefilename",
                "nproc",
                "hmmit",
                "atomtype",
                "selectatoms",
                "split",
                "miso",
                "timestep",
                "framesource",
                "inputfilename",
                "stepinterval",
                "printmoleculetime",
                "moleculeframes",
                "moleculetimesteps",
                "printreactionevent",
                "smilesworkmetrics",
            ],
            ["mname", "atomnames", "allmoleculeroute", "splitmoleculeroute"],
        )
        self._moleculeframefilter = self._getmoleculefilterarray(self.moleculeframes)
        self._moleculetimestepfilter = self._getmoleculefilterarray(
            self.moleculetimesteps
        )

    @staticmethod
    def getstype(rng):
        """Get a class for different methods.

        Following methonds are used to identify isomers:
        * SMILES (default)
        * VF2
        """
        if rng.SMILES:
            return _CollectSMILESPaths(rng)
        return _CollectMolPaths(rng)

    def collect(self):
        """Collect paths."""
        self.atomnames = self.atomname[self.atomtype]
        need_timed_output = self.printmoleculetime or self.printreactionevent
        if need_timed_output:
            started = time.perf_counter()
            store = TimedOutputStore(
                self.timedoutputfilename,
                cache_mib=self.timedoutputcachemib,
                input_filenames=self.inputfilename,
                timestep=self.timestep,
                frame_source=self.framesource,
                stepinterval=self.stepinterval,
                molecule_enabled=self.printmoleculetime,
                reaction_enabled=self.printreactionevent,
            )
            with store:
                self._collect(store)
                store.finalize_and_publish()
            database_seconds = store.write_seconds + store.finalize_seconds
            logger.info(
                "Timed-output core computation: %.3fs",
                max(0.0, time.perf_counter() - started - database_seconds),
            )
            return
        self._collect(None)

    def _collect(self, timed_store):
        stage_started = time.perf_counter()
        self._printmoleculename(timed_store)
        if timed_store is not None:
            timed_store.flush_molecules()
            if timed_store.molecule_enabled:
                logger.info(
                    "Timed-output molecule pipeline: %d ranges in %d batches, "
                    "%.3fs accounted, max %d definitions / %d ranges / %.3f MiB",
                    timed_store.molecule_range_count,
                    timed_store.molecule_write_batches,
                    timed_store.molecule_write_seconds,
                    timed_store.maximum_molecule_batch_definition_count,
                    timed_store.maximum_molecule_batch_range_count,
                    timed_store.maximum_molecule_batch_bytes / (1024 * 1024),
                )
        self._finish_step3_stage(timed_store, "molecule", stage_started)

        stage_started = time.perf_counter()
        matrix_store = self._getatomeach()
        try:
            matrix_store.save_molecule_names(self.mname)
            matrix_store.save_atom_types(self.atomtype)
            self._finish_step3_stage(timed_store, "matrix", stage_started)

            stage_started = time.perf_counter()
            self.allmoleculeroute = self._printatomroute(matrix_store)
            if self.split > 1:
                split_frames = np.array_split(np.arange(self.step), self.split)
                self.splitmoleculeroute = []
                for split_index, frames in enumerate(split_frames):
                    if len(frames) == 0:
                        frame_range = (0, 0)
                    else:
                        frame_range = (int(frames[0]), int(frames[-1]) + 1)
                    self.splitmoleculeroute.append(
                        self._printatomroute(
                            matrix_store,
                            timeaxis=split_index,
                            frame_range=frame_range,
                        )
                    )
            self.returnkeys()
            self._finish_step3_stage(timed_store, "route", stage_started)

            stage_started = time.perf_counter()
            ReactionsFinder(self.rng).findreactions(
                matrix_store,
                timed_store=timed_store,
                modified_atom_events=self._reaction_modified_atom_events,
                active_transitions=self._reaction_active_transitions,
            )
            self._finish_step3_stage(timed_store, "reaction", stage_started)
        finally:
            matrix_store.close()

    @staticmethod
    def _finish_step3_stage(timed_store, name, started):
        elapsed = max(0.0, time.perf_counter() - started)
        logger.info("Step 3 %s stage: %.3fs", name, elapsed)
        if timed_store is not None:
            timed_store.step3_stage_seconds[name] = elapsed

    @abstractmethod
    def _printmoleculename(self, timed_store):
        pass

    def _getatomeach(self):
        """Build disk-backed atom-frame matrices; molecule IDs start from 1."""
        molecule_dtype = self._molecule_index_dtype(self.hmmit)
        store = _AtomFrameStore((self.N, self.step), molecule_dtype)
        dense_frame_blocks = 0
        sparse_frame_blocks = 0
        empty_frame_blocks = 0
        dense_frame_values = 0
        direct_frame_molecules = 0
        direct_frame_values = 0
        molecule_fields = (0,) if self.runHMM else (0, 3)
        try:
            with (
                open(
                    self.hmmfilename if self.runHMM else self.originfilename, "rb"
                ) as fh,
                open(self.moleculetemp2filename, "rb") as ft,
            ):
                for molecule_id, (linehz, linetz) in enumerate(
                    tqdm(
                        _iter_atom_frame_records(
                            read_compressed_block(fh),
                            _iter_molecule_fields(ft, molecule_fields),
                            self.runHMM,
                            self.step,
                        ),
                        total=self.hmmit,
                        desc="Analyze atoms",
                        unit="molecule",
                        disable=None,
                    ),
                    start=1,
                ):
                    atoms = np.asarray(bytestolist(linetz[0]), dtype=np.int64)
                    if linehz is None:
                        frames = np.asarray(bytestolist(linetz[1])).reshape((-1,))
                        _write_indexed_atom_frames(
                            store,
                            atoms,
                            frames,
                            molecule_id,
                        )
                        direct_frame_molecules += 1
                        direct_frame_values += len(frames)
                        del atoms, frames
                        continue
                    signal = np.asarray(bytestolist(linehz)).reshape((-1,))
                    for frame_scan_start in range(
                        0,
                        len(signal),
                        _MATRIX_FRAME_SCAN_ROWS,
                    ):
                        frame_scan_stop = min(
                            frame_scan_start + _MATRIX_FRAME_SCAN_ROWS,
                            len(signal),
                        )
                        signal_block = signal[frame_scan_start:frame_scan_stop]
                        dense_block = _is_dense_signal_block(signal_block)
                        if dense_block:
                            dense_frame_blocks += 1
                            dense_frame_values += len(signal_block)
                            frames = None
                        else:
                            frames = np.flatnonzero(signal_block)
                            if frames.size == 0:
                                empty_frame_blocks += 1
                                del frames, signal_block
                                continue
                            sparse_frame_blocks += 1
                            frames += frame_scan_start
                        if dense_block:
                            _write_dense_atom_frame_span(
                                store,
                                atoms,
                                frame_scan_start,
                                frame_scan_stop,
                                molecule_id,
                            )
                        else:
                            _write_indexed_atom_frames(
                                store,
                                atoms,
                                frames,
                                molecule_id,
                            )
                        del frames, signal_block
                    del atoms, signal
            logger.info(
                "Atom-frame matrix input: %d no-HMM molecules / %d frame values "
                "read directly; signal blocks: %d dense / %d sparse / %d empty; "
                "%d dense frame values written without indices",
                direct_frame_molecules,
                direct_frame_values,
                dense_frame_blocks,
                sparse_frame_blocks,
                empty_frame_blocks,
                dense_frame_values,
            )
            store.flush()
            return store
        except BaseException:
            store.close()
            raise

    @staticmethod
    def _molecule_index_dtype(hmmit):
        """Return the smallest unsigned dtype that can store every molecule ID."""
        return _unsigned_dtype_for_maximum(max(1, int(hmmit)))

    def _getatomroute(self, item):
        i, (atomeachi, atomtypei) = item
        atomtype = np.asarray(self.atomtype).copy()
        atomtype[int(i) - 1] = atomtypei
        return _calculate_atom_route(
            int(i) - 1,
            atomeachi,
            atomtype,
            self.atomname,
            set(self.selectatoms),
            self.mname,
        )

    def _printatomroute(self, matrix_store, timeaxis=None, frame_range=None):
        """For analysis without HMM, we may not need to use np.unique."""
        if frame_range is None:
            frame_range = (0, self.step)
        frame_count = max(0, int(frame_range[1]) - int(frame_range[0]))
        active_transitions = None
        collect_active_transitions = (
            timeaxis is None
            and tuple(frame_range) == (0, self.step)
            and getattr(matrix_store, "active_transition_path", None) is not None
        )
        if collect_active_transitions:
            active_transitions = matrix_store.active_transitions
            active_transitions[:] = 0
            active_transitions.flush()
        route_nproc = _route_worker_count(self.nproc, self.N, frame_count)
        if route_nproc < self.nproc:
            logger.info(
                "Route worker count reduced from %d to %d for %d atom-frame values",
                self.nproc,
                route_nproc,
                self.N * frame_count,
            )
        assert matrix_store.mname_path is not None
        assert matrix_store.mname_values_path is not None
        assert matrix_store.atomtype_path is not None
        with WriteBuffer(
            open(
                (
                    self.atomroutefilename
                    if timeaxis is None
                    else f"{self.atomroutefilename}.{timeaxis}"
                ),
                "wb",
            ),
            sep=b"\n",
            byte_limit=_ROUTE_WRITE_BUFFER_BYTES,
        ) as f:
            species_route_counts = Counter()
            seen_molecule_routes = set()
            molecule_id_span = len(self.mname) + 1
            molecule_name_ids = self.mname.ids
            route_event_rows = 0
            route_pair_rows = 0
            compacted_atom_count = 0
            reaction_modified_atom_events = 0
            route_description = (
                "Collect reaction paths"
                if timeaxis is None
                else f"Collect reaction paths {timeaxis}"
            )
            if route_nproc == 1:
                start, stop = frame_range
                atomtype = np.asarray(self.atomtype)
                selectatoms = set(self.selectatoms)
                results = (
                    _get_atom_route_result(
                        atom_index,
                        matrix_store.atomeach[atom_index, start:stop],
                        atomtype,
                        self.atomname,
                        selectatoms,
                        self.mname,
                        active_transitions,
                    )
                    for atom_index in tqdm(
                        range(self.N),
                        total=self.N,
                        desc=route_description,
                        unit="atom",
                        disable=None,
                    )
                )
            else:
                results = run_mp(
                    route_nproc,
                    func=_get_atom_route_by_index,
                    l=range(self.N),
                    unordered=False,
                    chunksize=1,
                    max_inflight=max(2, 2 * route_nproc),
                    disk_ordered=True,
                    initializer=_initialize_route_worker,
                    initargs=(
                        matrix_store.atomeach_path,
                        matrix_store.shape,
                        matrix_store.molecule_dtype.str,
                        matrix_store.mname_path,
                        matrix_store.mname_values_path,
                        matrix_store.atomtype_path,
                        self.atomname,
                        tuple(self.selectatoms),
                        tuple(frame_range),
                        (
                            matrix_store.active_transition_path
                            if collect_active_transitions
                            else None
                        ),
                        frame_count - 1 if collect_active_transitions else 0,
                    ),
                    maxtasksperchild=None,
                    total=self.N,
                    desc=route_description,
                    unit="atom",
                )
            for (
                moleculeroute,
                route_counts,
                route_payload,
                modified_atom_events,
            ) in results:
                f.append(route_payload)
                reaction_modified_atom_events += int(modified_atom_events)
                route_pair_rows += len(moleculeroute)
                if route_counts is None:
                    route_event_rows += len(moleculeroute)
                else:
                    route_event_rows += int(route_counts.sum(dtype=np.uint64))
                    compacted_atom_count += 1
                current_atom_routes = set() if not self.runHMM else None
                if moleculeroute.size > 0:
                    if route_counts is not None and len(route_counts) != len(
                        moleculeroute
                    ):
                        raise RuntimeError("Route pair counts do not match route pairs")
                    for route_index, rr in enumerate(moleculeroute):
                        left = int(rr[0])
                        right = int(rr[1])
                        route_key = left * molecule_id_span + right
                        if not self.runHMM:
                            # check whether repeated or not if analyzing without HMM
                            if route_key in seen_molecule_routes:
                                continue
                            assert current_atom_routes is not None
                            current_atom_routes.add(route_key)
                        else:
                            if route_key in seen_molecule_routes:
                                continue
                            seen_molecule_routes.add(route_key)
                        species_pair = (
                            int(molecule_name_ids[left - 1]),
                            int(molecule_name_ids[right - 1]),
                        )
                        if species_pair[0] != species_pair[1]:
                            species_route_counts[species_pair] += (
                                1
                                if self.runHMM or route_counts is None
                                else int(route_counts[route_index])
                            )
                if current_atom_routes is not None:
                    seen_molecule_routes.update(current_atom_routes)
        logger.info(
            "Route worker compaction: %d event rows -> %d pair rows, "
            "%d compacted atoms",
            route_event_rows,
            route_pair_rows,
            compacted_atom_count,
        )
        logger.info(
            "Route output buffer: %.3f MiB maximum encoded batch "
            "(%.3f MiB target plus one indivisible line)",
            f.maximum_buffer_bytes / (1024 * 1024),
            _ROUTE_WRITE_BUFFER_BYTES / (1024 * 1024),
        )
        logger.info(
            "Route aggregation%s: %d unique molecule pairs, %d species pairs",
            "" if timeaxis is None else f" {timeaxis}",
            len(seen_molecule_routes),
            len(species_route_counts),
        )
        if timeaxis is None:
            self._reaction_modified_atom_events = reaction_modified_atom_events
            if active_transitions is None:
                self._reaction_active_transitions = None
            else:
                active_transitions.flush()
                self._reaction_active_transitions = np.flatnonzero(active_transitions)
            logger.info(
                "Reaction scheduling input: %d modified atom events, %s active "
                "transitions",
                reaction_modified_atom_events,
                (
                    "unknown"
                    if self._reaction_active_transitions is None
                    else str(len(self._reaction_active_transitions))
                ),
            )
        return species_route_counts

    def _re(self, smi):
        """If you use RDkit to convert a methyl radical to SMILES, you will get something
        like [H]C([H])[H]. However, OpenBabel will consider it as a methane molecule. So,
        you have to use [H][C]([H])[H], if you need to process some radicals.

        Examples
        --------
        >>> self._re('C')
        [C]
        >>> self._re('[C]')
        [C]
        >>> self._re('[CH]')
        [CH]
        >>> self._re('Na')
        [Na]
        >>> self._re('[H]c(Cl)C([H])Cl')
        [H][c]([Cl])[C]([H])[Cl]
        """
        if "_unknownSMILES" in smi:
            # not SMILES
            return smi
        pattern = getattr(self, "_compiled_smiles_element_pattern", None)
        if pattern is None:
            sorted_atoms = sorted(
                self.atomname,
                key=lambda atom: len(atom),
                reverse=True,
            )
            elements = "|".join(
                [
                    ((atom.upper() + "|" + atom.lower()) if len(atom) == 1 else atom)
                    for atom in sorted_atoms
                    if atom != "H"
                ]
            )
            pattern = re.compile(r"(?<!\[)(" + elements + r")(?!H|\])")
            self._compiled_smiles_element_pattern = pattern
        smi = pattern.sub(r"[\1]", smi)
        return smi.replace("[HH]", "[H]")

    def convertSMILES(self, atoms, bonds):
        """Convert atoms and bonds information to SMILES.

        Raises
        ------
        ValueError
            (RDKit error) Maximum BFS search size exceeded.
        """
        empty_molecule = getattr(self, "_empty_rdkit_molecule", None)
        if empty_molecule is None:
            empty_molecule = Chem.MolFromSmiles("")
            self._empty_rdkit_molecule = empty_molecule
        m = Chem.RWMol(empty_molecule)
        d = {}
        for name, number in zip(self.atomnames[atoms], atoms):
            d[number] = m.AddAtom(Chem.Atom(name))
        for atom1, atom2, level in bonds:
            m.AddBond(d[atom1], d[atom2], Chem.BondType(level))
        # https://github.com/rdkit/rdkit/discussions/6613#discussioncomment-6688021
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
        name = Chem.MolToSmiles(m)
        return self._re(name)

    def _getatomsandbonds(self, line):
        atoms = np.array(bytestolist(line[0]), dtype=np.uint64)
        pairs = bytestolist(line[1])
        levels = bytestolist(line[2])
        bonds = [[*pair, level] for pair, level in zip(pairs, levels)]
        return atoms, bonds

    def _getmoleculeframes(self, line):
        return np.asarray(bytestolist(line[-1]))

    def _needmoleculetimeline(self):
        return (
            self.printmoleculetime
            or self._moleculeframefilter is not None
            or self._moleculetimestepfilter is not None
        )

    def _getmoleculetimesteps(self, frames):
        return [get_timestep_value(self.timestep[int(frame)]) for frame in frames]

    @staticmethod
    def _hasmoleculefilter(values):
        return values is not None and len(values) > 0

    @classmethod
    def _getmoleculefilterarray(cls, values):
        if not cls._hasmoleculefilter(values):
            return None
        return np.asarray(values, dtype=np.int64).reshape((-1,))

    def _formatmoleculename(self, name, atoms, bonds):
        atom_text = ";".join(map(str, atoms))
        bond_text = ";".join(",".join(map(str, bond)) for bond in bonds)
        return " ".join((name, atom_text, bond_text))

    @staticmethod
    def _formatmoleculeatomids(atoms):
        return ";".join(str(atom) for atom in atoms)

    @staticmethod
    def _formatmoleculebondids(bonds):
        return ";".join("-".join(str(item) for item in bond) for bond in bonds)

    def _itermoleculeranges(self, frames):
        frame_values = np.asarray(frames).reshape((-1,))
        if frame_values.size == 0:
            return
        if (
            frame_values.size <= _RANGE_LINEAR_SCAN_ROWS
            and self._moleculeframefilter is None
            and self._moleculetimestepfilter is None
        ):
            if frame_values.dtype.kind not in "iu":
                frame_values = frame_values.astype(np.int64)
            range_starts = []
            range_ends = []
            pending_start = int(frame_values[0])
            pending_end = pending_start
            for frame in frame_values[1:]:
                frame = int(frame)
                if frame - pending_end in (0, 1):
                    pending_end = frame
                    continue
                range_starts.append(pending_start)
                range_ends.append(pending_end)
                pending_start = frame
                pending_end = frame
            range_starts.append(pending_start)
            range_ends.append(pending_end)
            yield (
                np.asarray(range_starts, dtype=np.uint64),
                np.asarray(range_ends, dtype=np.uint64),
            )
            return
        pending_start = None
        pending_end = None
        for block_start in range(0, len(frame_values), _RANGE_SCAN_ROWS):
            selected_frames = frame_values[block_start : block_start + _RANGE_SCAN_ROWS]
            if not np.issubdtype(selected_frames.dtype, np.integer):
                selected_frames = selected_frames.astype(np.int64)
            if self._moleculeframefilter is not None:
                selected_frames = selected_frames[
                    np.isin(selected_frames, self._moleculeframefilter)
                ]
            if self._moleculetimestepfilter is not None and selected_frames.size:
                selected_timesteps = np.fromiter(
                    (
                        int(get_timestep_value(self.timestep[int(frame)]))
                        for frame in selected_frames
                    ),
                    dtype=np.int64,
                    count=len(selected_frames),
                )
                selected_frames = selected_frames[
                    np.isin(selected_timesteps, self._moleculetimestepfilter)
                ]
            if selected_frames.size == 0:
                continue

            frame_deltas = np.diff(selected_frames)
            range_breaks = np.flatnonzero((frame_deltas != 0) & (frame_deltas != 1))
            starts = np.concatenate(
                (selected_frames[:1], selected_frames[range_breaks + 1])
            )
            ends = np.concatenate((selected_frames[range_breaks], selected_frames[-1:]))
            if pending_start is not None:
                gap = int(starts[0]) - pending_end
                if gap in (0, 1):
                    starts[0] = pending_start
                else:
                    starts = np.concatenate(
                        (np.asarray([pending_start], dtype=starts.dtype), starts)
                    )
                    ends = np.concatenate(
                        (np.asarray([pending_end], dtype=ends.dtype), ends)
                    )

            pending_start = int(starts[-1])
            pending_end = int(ends[-1])
            ready_starts = starts[:-1]
            ready_ends = ends[:-1]
            for output_start in range(
                0,
                len(ready_starts),
                _RANGE_OUTPUT_BLOCK_ROWS,
            ):
                output_stop = output_start + _RANGE_OUTPUT_BLOCK_ROWS
                yield (
                    ready_starts[output_start:output_stop],
                    ready_ends[output_start:output_stop],
                )

        if pending_start is not None:
            yield (
                np.asarray([pending_start], dtype=np.uint64),
                np.asarray([pending_end], dtype=np.uint64),
            )

    def _storetimedmolecule(
        self,
        timed_store,
        molecule_id,
        name,
        atoms,
        bonds,
        frames,
    ):
        if timed_store is None or not self._needmoleculetimeline():
            return
        timed_store.add_molecule(
            molecule_id,
            name,
            atoms,
            bonds,
            self._itermoleculeranges(frames),
        )

    def _finishmoleculenames(self, builder: _MoleculeNameBuilder) -> None:
        self.mname = builder.finish()
        logger.info(
            "Compact molecule names: %d molecules, %d species, %.3f MiB",
            len(self.mname),
            len(self.mname.names),
            (self.mname.ids.nbytes + self.mname.names.nbytes) / (1024 * 1024),
        )


class _CollectMolPaths(_CollectPaths):
    """VF2 is used to identify isomers.

    If SMILES is failed to generate, fallback to the name like CxHyOz.
    """

    def _printmoleculename(self, timed_store):
        mname = _MoleculeNameBuilder(self.hmmit)
        d = defaultdict(list)
        em = iso.numerical_edge_match(["atom", "level"], ["None", 1])
        # idx for unknown SMILES
        self.n_unknown = 0
        with (
            WriteBuffer(open(self.moleculefilename, "w"), sep="\n") as fm,
            open(self.moleculetemp2filename, "rb") as ft,
        ):
            lines = itertools.zip_longest(*[read_compressed_block(ft)] * 4)
            for molecule_id, line in enumerate(
                tqdm(
                    lines,
                    total=self.hmmit,
                    desc="Indentify isomers",
                    unit="molecule",
                    disable=None,
                ),
                start=1,
            ):
                atoms, bonds = self._getatomsandbonds(line)
                molecule = Molecule(self, atoms, bonds)
                for isomer in d[str(molecule)]:
                    if isomer.isomorphic(molecule, em):
                        molecule.smiles = isomer.smiles
                        break
                else:
                    d[str(molecule)].append(molecule)
                mname.append(molecule.smiles)
                fm.append(self._formatmoleculename(molecule.smiles, atoms, bonds))
                if self._needmoleculetimeline():
                    self._storetimedmolecule(
                        timed_store,
                        molecule_id,
                        molecule.smiles,
                        atoms,
                        bonds,
                        self._getmoleculeframes(line),
                    )
        self._finishmoleculenames(mname)


class _CollectSMILESPaths(_CollectPaths):
    def _smiles_structure_cache_key(self, atoms, bonds):
        """Encode one exactly labelled graph independently of global atom IDs."""
        atom_ids = np.asarray(atoms).reshape((-1,))
        key_payload_bytes = (
            atom_ids.size * _SMILES_NAME_CACHE_ATOM_BYTES
            + len(bonds) * _SMILES_NAME_CACHE_BOND_BYTES
        )
        if (
            key_payload_bytes + _SMILES_NAME_CACHE_ENTRY_OVERHEAD
            > _SMILES_NAME_CACHE_BYTES
        ):
            return None
        atom_types = np.asarray(self.atomtype[atom_ids], dtype=np.int64)
        atom_positions = {
            int(atom_id): position for position, atom_id in enumerate(atom_ids)
        }
        local_bonds = np.empty((len(bonds), 3), dtype=np.int64)
        for row, (atom1, atom2, level) in enumerate(bonds):
            local_bonds[row] = (
                atom_positions[int(atom1)],
                atom_positions[int(atom2)],
                int(level),
            )
        return atom_types.tobytes(), local_bonds.tobytes()

    def _get_cached_smiles_name(self, key):
        cache = getattr(self, "_smiles_name_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._smiles_name_cache = cache
            self._smiles_name_cache_bytes = 0
            return False, None
        try:
            value = cache.pop(key)
        except KeyError:
            return False, None
        cache[key] = value
        return True, value[0]

    def _cache_smiles_name(self, key, name):
        cache = self._smiles_name_cache
        name_bytes = len(name.encode("utf-8")) if name is not None else 0
        entry_bytes = (
            len(key[0]) + len(key[1]) + name_bytes + _SMILES_NAME_CACHE_ENTRY_OVERHEAD
        )
        if entry_bytes > _SMILES_NAME_CACHE_BYTES:
            return
        while (
            cache
            and self._smiles_name_cache_bytes + entry_bytes > _SMILES_NAME_CACHE_BYTES
        ):
            _, (_, removed_bytes) = cache.popitem(last=False)
            self._smiles_name_cache_bytes -= removed_bytes
        cache[key] = (name, entry_bytes)
        self._smiles_name_cache_bytes += entry_bytes

    def _printmoleculename(self, timed_store):
        mname = _MoleculeNameBuilder(self.hmmit)
        d = defaultdict(list)
        name_mapping = {}
        name_mapping_graph = defaultdict(dict)
        em = iso.numerical_edge_match(["atom", "level"], ["None", 1])
        self.n_unknown = 0
        need_molecule_timeline = self._needmoleculetimeline()
        work_metrics = getattr(self, "smilesworkmetrics", None)
        structure_bytes = (
            work_metrics.total_compressed_bytes if work_metrics is not None else None
        )
        maximum_structure_bytes = (
            work_metrics.max_compressed_record_bytes
            if work_metrics is not None
            else None
        )
        smiles_start_method = get_start_method()
        smiles_nproc = _smiles_worker_count(
            self.nproc,
            self.hmmit,
            structure_bytes,
            smiles_start_method,
        )
        smiles_chunksize, smiles_max_inflight = _smiles_pool_limits(
            smiles_nproc,
            self.hmmit,
            structure_bytes,
            smiles_start_method,
        )
        reuse_worker_structures = _should_reuse_smiles_worker_structures(
            smiles_nproc,
            smiles_chunksize,
            smiles_start_method,
            maximum_structure_bytes,
        )
        if smiles_nproc < self.nproc:
            average_structure_bytes = (
                int(structure_bytes) / self.hmmit
                if structure_bytes is not None and self.hmmit
                else 0.0
            )
            logger.info(
                "SMILES worker count reduced from %d to %d for %d molecules "
                "and %.1f average compressed structure bytes using %s start",
                self.nproc,
                smiles_nproc,
                self.hmmit,
                average_structure_bytes,
                smiles_start_method,
            )
        if smiles_nproc > 1:
            logger.info(
                "SMILES pool: %d workers, chunksize=%d, max_inflight=%d",
                smiles_nproc,
                smiles_chunksize,
                smiles_max_inflight,
            )
            if reuse_worker_structures:
                logger.info(
                    "SMILES parent decode: reusing worker structures and reading "
                    "only timeline payloads; maximum compressed structure "
                    "bytes=%d (reuse limit=%d)",
                    int(maximum_structure_bytes),
                    _SMILES_FORK_REUSE_MAX_STRUCTURE_BYTES,
                )
            elif smiles_chunksize > 1:
                logger.info(
                    "SMILES parent decode: retaining name-only worker results for "
                    "maximum compressed structure bytes=%s (reuse limit=%d)",
                    (
                        "unknown"
                        if maximum_structure_bytes is None
                        else str(int(maximum_structure_bytes))
                    ),
                    _SMILES_FORK_REUSE_MAX_STRUCTURE_BYTES,
                )
        with (
            WriteBuffer(open(self.moleculefilename, "w"), sep="\n") as fm,
            open(self.moleculetemp2filename, "rb") as ft,
            open(self.moleculetemp2filename, "rb") as parent_ft,
        ):
            if smiles_nproc == 1:
                selected_parent_fields = (
                    (0, 1, 2, 3) if need_molecule_timeline else (0, 1, 2)
                )
                parent_fields = _iter_molecule_fields(
                    parent_ft,
                    selected_parent_fields,
                )
                result_records = (
                    self._get_serial_smiles_record(record)
                    for record in tqdm(
                        parent_fields,
                        total=self.hmmit,
                        desc="Indentify isomers",
                        unit="molecule",
                        disable=None,
                    )
                )
            else:
                worker_lines = _iter_molecule_fields(ft, (0, 1, 2))
                results = run_mp(
                    smiles_nproc,
                    func=(
                        _get_smiles_name_and_structure
                        if reuse_worker_structures
                        else _get_smiles_name
                    ),
                    l=worker_lines,
                    unordered=False,
                    chunksize=smiles_chunksize,
                    max_inflight=smiles_max_inflight,
                    disk_ordered=True,
                    initializer=_initialize_smiles_worker,
                    initargs=(self.atomname, self.atomtype),
                    maxtasksperchild=None,
                    total=self.hmmit,
                    desc="Indentify isomers",
                    unit="molecule",
                )
                if reuse_worker_structures:
                    parent_frames = (
                        _iter_molecule_fields(parent_ft, (3,))
                        if need_molecule_timeline
                        else None
                    )
                    result_records = (
                        _SmilesResultRecord.from_decoded(
                            name,
                            atoms,
                            bonds,
                            (
                                next(parent_frames)[0]
                                if parent_frames is not None
                                else None
                            ),
                        )
                        for name, atoms, bonds in results
                    )
                else:
                    selected_parent_fields = (
                        (0, 1, 2, 3) if need_molecule_timeline else (0, 1, 2)
                    )
                    parent_fields = _iter_molecule_fields(
                        parent_ft,
                        selected_parent_fields,
                    )
                    result_records = (
                        _SmilesResultRecord.from_compressed(
                            name,
                            next(parent_fields),
                        )
                        for name in results
                    )
            for molecule_id, result in enumerate(result_records, start=1):
                name = result.name
                atoms = result.atoms
                bonds = result.bonds
                if atoms is None:
                    assert bonds is None
                    assert result.structure_record is not None
                    atoms, bonds = self._getatomsandbonds(result.structure_record)
                if name is None:
                    # SMILES failed, fallback to VF2 identify isomers
                    molecule = Molecule(self, atoms, bonds)

                    # directly raise ValueError to save time
                    def _raise_anyway(*args, **kwargs):
                        raise ValueError("Maximum BFS search size exceeded.")

                    molecule._convertSMILES = _raise_anyway
                    for isomer in d[str(molecule)]:
                        if isomer.isomorphic(molecule, em):
                            molecule.smiles = isomer.smiles
                            break
                    else:
                        d[str(molecule)].append(molecule)
                    name = molecule.smiles
                if self.miso > 0:
                    if name in name_mapping:
                        name = name_mapping[name]
                    else:
                        # check if the name is isomorphic to the previous molecules
                        molecule = Molecule(self, atoms, bonds)
                        # the formula should be the same
                        mng = name_mapping_graph[molecule.name]
                        for isomer, mol in mng.items():
                            if mol.isomorphic(molecule, em):
                                # use the previous SMILES
                                name_mapping[name] = isomer
                                name = isomer
                                break
                        else:
                            mng[name] = molecule
                            name_mapping[name] = name
                mname.append(name)
                fm.append(self._formatmoleculename(name, atoms, bonds))
                if need_molecule_timeline:
                    assert result.frame_block is not None
                    self._storetimedmolecule(
                        timed_store,
                        molecule_id,
                        name,
                        atoms,
                        bonds,
                        bytestolist(result.frame_block),
                    )
        self._finishmoleculenames(mname)

    def _get_serial_smiles_record(self, record):
        """Decode one parent record once for SMILES and downstream output."""
        atoms, bonds = self._getatomsandbonds(record)
        return _SmilesResultRecord.from_decoded(
            self._calmoleculeSMILESname_from_decoded(atoms, bonds),
            atoms,
            bonds,
            record[3] if len(record) > 3 else None,
        )

    def _calmoleculeSMILESname(self, item):
        atoms, bonds = self._getatomsandbonds(item)
        return self._calmoleculeSMILESname_from_decoded(atoms, bonds)

    def _calmoleculeSMILESname_from_decoded(self, atoms, bonds):
        """Convert already decoded structure arrays to one species name."""
        cache_key = self._smiles_structure_cache_key(atoms, bonds)
        if cache_key is not None:
            found, name = self._get_cached_smiles_name(cache_key)
            if found:
                return name
        try:
            name = self.convertSMILES(atoms, bonds)
        except ValueError:
            # fallback to VF2
            name = None
        if cache_key is not None:
            self._cache_smiles_name(cache_key, name)
        return name


class Molecule:
    """A molecule class for isomer identification."""

    def __init__(self, cmp, atoms, bonds):
        self.cmp = cmp
        self.atoms = atoms
        self.bonds = bonds
        self._atomtypes = cmp.atomtype[atoms]
        self._atomnames = cmp.atomnames[atoms]
        self._miso = cmp.miso
        self.graph = self._makemoleculegraph()
        counter = Counter(self._atomnames)
        self.name = "".join(
            f"{atomname}{counter[atomname]}" for atomname in cmp.atomname
        )
        self._smiles = None
        self._convertSMILES = cmp.convertSMILES

    def __str__(self):
        return self.name

    @property
    def smiles(self):
        """Return SMILES of a molecule."""
        if self._smiles is None:
            try:
                self._smiles = self._convertSMILES(self.atoms, self.bonds)
            except ValueError:
                # when RDKit error: Maximum BFS search size exceeded
                # fallback to the name of the molecule
                # blank should be avoided
                self._smiles = self.name + f"_unknownSMILES_{self.cmp.n_unknown}"
                self.cmp.n_unknown += 1
        return self._smiles

    @smiles.setter
    def smiles(self, value):
        self._smiles = value

    def _makemoleculegraph(self):
        graph = nx.Graph()
        for line in self.bonds:
            if self._miso == 0:
                # normal mode
                graph.add_edge(line[0], line[1], level=line[2])
            elif self._miso == 1:
                # merge the isomers with same atoms and same bond-network but different bond orders
                graph.add_edge(line[0], line[1], level=1)
            elif self._miso == 2:
                # merge the isomers with same atoms with different bond-network
                pass
            else:
                raise ValueError(f"Unknown isomer identification method: {self._miso}.")
        for atomnumber, atomtype in zip(self.atoms, self._atomtypes):
            graph.add_node(atomnumber, atom=atomtype)
        return graph

    def isomorphic(self, mol, em):
        """Return whether two molecules are isomorphic."""
        return nx.is_isomorphic(self.graph, mol.graph, em)
