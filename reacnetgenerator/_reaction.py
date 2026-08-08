# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Reactions finder."""

from collections import Counter, defaultdict
from multiprocessing import get_start_method

import numpy as np
from tqdm.auto import tqdm

from ._logging import logger
from ._moleculenames import _MoleculeNameTable
from ._packedbool import _PackedBoolMatrix
from .dps import dps_reaction  # type:ignore
from .utils import SharedRNGData, WriteBuffer, bytestolist, run_mp

_REACTION_ATOMEACH = None
_REACTION_CONFLICT = None
_REACTION_MNAME = None
_REACTION_ATOM_SCAN_ROWS = 65536
_REACTION_NEIGHBOR_DICT_THRESHOLD = 8
_REACTION_PAIR_COMPACTION_BLOCK_ROWS = 1024
_REACTION_PAIR_COMPACTION_MIN_ROWS = 256
_REACTION_PAIR_COMPACTION_SAMPLE_ROWS = 1024
_REACTION_FORK_CHANGED_ATOM_EVENTS_PER_WORKER = 50_000
_REACTION_CHANGED_ATOM_EVENTS_PER_WORKER = 1_000_000
_REACTION_SCAN_VALUES_PER_WORKER = 500_000_000
_REACTION_MAX_CHUNKSIZE = 32
_REACTION_MIN_CHUNKS_PER_WORKER = 4
_REACTION_CHANGED_ATOM_EVENTS_PER_CHUNK = 4096
_REACTION_SCAN_VALUES_PER_CHUNK = 1_000_000


def _add_reaction_neighbor(mapping, molecule_id, neighbor_id):
    """Append one unique neighbor while preserving deterministic DFS order."""
    neighbors = mapping[molecule_id]
    if isinstance(neighbors, dict):
        neighbors.setdefault(neighbor_id, None)
        return
    if neighbor_id in neighbors:
        return
    if len(neighbors) >= _REACTION_NEIGHBOR_DICT_THRESHOLD:
        mapping[molecule_id] = dict.fromkeys((*neighbors, neighbor_id))
    else:
        neighbors.append(neighbor_id)


def _reaction_pair_first_indices(before_values, after_values):
    """Return first occurrences of molecule pairs in original row order."""
    if len(before_values) == 0:
        return np.zeros(0, dtype=np.intp)
    uint32_max = np.iinfo(np.uint32).max
    minimum = min(
        int(np.min(before_values, initial=0)),
        int(np.min(after_values, initial=0)),
    )
    maximum = max(
        int(np.max(before_values, initial=0)),
        int(np.max(after_values, initial=0)),
    )
    if minimum >= 0 and maximum <= uint32_max:
        pair_keys = np.asarray(before_values, dtype=np.uint64).copy()
        np.left_shift(pair_keys, np.uint64(32), out=pair_keys)
        np.bitwise_or(
            pair_keys,
            np.asarray(after_values, dtype=np.uint64),
            out=pair_keys,
        )
        first_indices = np.unique(pair_keys, return_index=True)[1]
    else:
        pairs = np.column_stack((before_values, after_values))
        first_indices = np.unique(pairs, axis=0, return_index=True)[1]
    first_indices.sort()
    return first_indices


def _reaction_pair_run_first_indices(
    before_values,
    after_values,
    run_starts_buffer=None,
    after_changes_buffer=None,
):
    """Return first rows of consecutive equal pair runs in original order."""
    if len(before_values) == 0:
        return np.zeros(0, dtype=np.intp)
    if run_starts_buffer is None:
        run_starts = np.empty(len(before_values), dtype=np.bool_)
    else:
        run_starts = run_starts_buffer[: len(before_values)]
    run_starts[0] = True
    np.not_equal(before_values[1:], before_values[:-1], out=run_starts[1:])
    after_change_count = max(0, len(after_values) - 1)
    if after_changes_buffer is None:
        after_changes = np.empty(after_change_count, dtype=np.bool_)
    else:
        after_changes = after_changes_buffer[:after_change_count]
    np.not_equal(after_values[1:], after_values[:-1], out=after_changes)
    np.logical_or(run_starts[1:], after_changes, out=run_starts[1:])
    return np.flatnonzero(run_starts)


def _should_compact_reaction_pairs(before_block, after_block, modified_atoms):
    """Use full pair compaction only when a bounded sample is highly repetitive."""
    if len(modified_atoms) < _REACTION_PAIR_COMPACTION_MIN_ROWS:
        return False
    sample_step = max(
        1,
        len(modified_atoms) // _REACTION_PAIR_COMPACTION_SAMPLE_ROWS,
    )
    sample_atoms = modified_atoms[::sample_step][:_REACTION_PAIR_COMPACTION_SAMPLE_ROWS]
    unique_count = len(
        _reaction_pair_first_indices(
            before_block[sample_atoms],
            after_block[sample_atoms],
        )
    )
    return unique_count * 4 <= len(sample_atoms) * 3


def _add_compact_conflict_neighbors(mapping, molecule_values, conflict_mask):
    """Add one conflict marker per molecule while retaining first-seen order."""
    conflict_molecules = molecule_values[conflict_mask]
    if len(conflict_molecules) == 0:
        return
    first_indices = np.unique(conflict_molecules, return_index=True)[1]
    first_indices.sort()
    for first_index in first_indices:
        _add_reaction_neighbor(
            mapping,
            int(conflict_molecules[first_index]),
            ReactionsFinder.CONFLICT,
        )


def _reaction_worker_count(
    requested_nproc,
    atom_count,
    transition_count,
    modified_atom_events,
    start_method=None,
):
    """Amortize platform-specific process startup over observed work."""
    requested_nproc = int(requested_nproc)
    atom_count = max(0, int(atom_count))
    transition_count = max(0, int(transition_count))
    maximum_workers = min(requested_nproc, max(1, transition_count))
    if modified_atom_events is None:
        return maximum_workers
    if start_method is None:
        start_method = get_start_method()
    event_target = (
        _REACTION_FORK_CHANGED_ATOM_EVENTS_PER_WORKER
        if start_method == "fork"
        else _REACTION_CHANGED_ATOM_EVENTS_PER_WORKER
    )
    modified_atom_events = max(0, int(modified_atom_events))
    event_workers = max(
        1,
        modified_atom_events // event_target,
    )
    scan_workers = max(
        1,
        (atom_count * transition_count) // _REACTION_SCAN_VALUES_PER_WORKER,
    )
    return min(maximum_workers, max(event_workers, scan_workers))


def _reaction_chunksize(
    worker_count,
    atom_count,
    transition_count,
    modified_atom_events,
):
    """Batch cheap transitions while bounding work and results per IPC payload."""
    worker_count = max(1, int(worker_count))
    atom_count = max(0, int(atom_count))
    transition_count = max(0, int(transition_count))
    if worker_count == 1 or transition_count == 0 or modified_atom_events is None:
        return 1
    modified_atom_events = max(0, int(modified_atom_events))
    average_modified_events = max(
        1,
        (modified_atom_events + transition_count - 1) // transition_count,
    )
    return min(
        _REACTION_MAX_CHUNKSIZE,
        max(
            1,
            transition_count // (worker_count * _REACTION_MIN_CHUNKS_PER_WORKER),
        ),
        max(
            1,
            _REACTION_SCAN_VALUES_PER_CHUNK // max(1, atom_count),
        ),
        max(
            1,
            _REACTION_CHANGED_ATOM_EVENTS_PER_CHUNK // average_modified_events,
        ),
    )


def _normalize_active_transitions(active_transitions, transition_count):
    """Return sorted unique transition indices or ``None`` when unknown."""
    if active_transitions is None:
        return None
    transition_count = max(0, int(transition_count))
    values = np.asarray(active_transitions)
    if values.ndim != 1:
        values = values.reshape((-1,))
    if values.dtype == np.bool_:
        if len(values) != transition_count:
            raise ValueError("Active transition mask length does not match trajectory")
        return np.flatnonzero(values)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    if values.dtype.kind not in "iu":
        raise TypeError("Active transitions must contain integer indices")
    if np.any(values < 0) or np.any(values >= transition_count):
        raise ValueError("Active transition index is outside the trajectory")
    normalized = values.astype(np.int64, copy=False)
    if len(normalized) > 1 and np.any(normalized[1:] <= normalized[:-1]):
        normalized = np.unique(normalized)
    return normalized


def _initialize_reaction_worker(
    atomeach_path,
    conflict_path,
    shape,
    molecule_dtype_string,
    mname_path,
    mname_values_path,
):
    """Attach one reaction worker to read-only shared mappings."""
    global _REACTION_ATOMEACH
    global _REACTION_CONFLICT
    global _REACTION_MNAME
    _REACTION_ATOMEACH = np.memmap(
        atomeach_path,
        mode="r",
        dtype=np.dtype(molecule_dtype_string),
        shape=tuple(shape),
    )
    _REACTION_CONFLICT = _PackedBoolMatrix(
        conflict_path,
        shape,
        mode="r",
    )
    _REACTION_MNAME = _MoleculeNameTable(
        np.load(mname_path, mmap_mode="r", allow_pickle=False),
        np.load(mname_values_path, mmap_mode="r", allow_pickle=False),
        validate=False,
    )


def _filter_reaction_pair(reaction, molecule_names):
    leftname, rightname = (
        Counter(str(molecule_names[int(index) - 1]) for index in side)
        for side in reaction
    )
    new_leftname = leftname - rightname
    new_rightname = rightname - leftname
    if new_leftname and new_rightname:
        return tuple(
            "+".join(sorted(side.elements())) for side in (new_leftname, new_rightname)
        )
    return None


def _calculate_transition_reactions(
    atomeach_before,
    atomeach_after,
    conflict_before,
    conflict_after,
    molecule_names,
):
    """Return a compact counter for one adjacent-frame transition."""
    reactdict = [defaultdict(list), defaultdict(list)]
    run_starts_buffer = None
    after_changes_buffer = None
    for atom_start in range(0, len(atomeach_before), _REACTION_ATOM_SCAN_ROWS):
        atom_stop = min(atom_start + _REACTION_ATOM_SCAN_ROWS, len(atomeach_before))
        before_block = atomeach_before[atom_start:atom_stop]
        after_block = atomeach_after[atom_start:atom_stop]
        modified_atoms = np.flatnonzero(before_block != after_block)
        if _should_compact_reaction_pairs(
            before_block,
            after_block,
            modified_atoms,
        ):
            conflict_blocks = (
                conflict_before[atom_start:atom_stop],
                conflict_after[atom_start:atom_stop],
            )
            conflict_present = tuple(np.any(values) for values in conflict_blocks)
            if run_starts_buffer is None:
                run_starts_buffer = np.empty(
                    _REACTION_PAIR_COMPACTION_BLOCK_ROWS,
                    dtype=np.bool_,
                )
                after_changes_buffer = np.empty(
                    _REACTION_PAIR_COMPACTION_BLOCK_ROWS - 1,
                    dtype=np.bool_,
                )
            for compact_start in range(
                0,
                len(modified_atoms),
                _REACTION_PAIR_COMPACTION_BLOCK_ROWS,
            ):
                compact_atoms = modified_atoms[
                    compact_start : compact_start + _REACTION_PAIR_COMPACTION_BLOCK_ROWS
                ]
                before_values = before_block[compact_atoms]
                after_values = after_block[compact_atoms]
                first_indices = _reaction_pair_run_first_indices(
                    before_values,
                    after_values,
                    run_starts_buffer,
                    after_changes_buffer,
                )
                if len(first_indices) * 4 > len(compact_atoms) * 3:
                    first_indices = _reaction_pair_first_indices(
                        before_values,
                        after_values,
                    )
                for first_index in first_indices:
                    before = int(before_values[first_index])
                    after = int(after_values[first_index])
                    _add_reaction_neighbor(reactdict[0], before, after)
                    _add_reaction_neighbor(reactdict[1], after, before)
                if conflict_present[0]:
                    _add_compact_conflict_neighbors(
                        reactdict[0],
                        before_values,
                        conflict_blocks[0][compact_atoms],
                    )
                if conflict_present[1]:
                    _add_compact_conflict_neighbors(
                        reactdict[1],
                        after_values,
                        conflict_blocks[1][compact_atoms],
                    )
        else:
            changed_atom_indices = atom_start + modified_atoms
            before_conflicts = conflict_before[changed_atom_indices]
            after_conflicts = conflict_after[changed_atom_indices]
            for changed_index, local_atom_index in enumerate(modified_atoms):
                before = int(before_block[local_atom_index])
                after = int(after_block[local_atom_index])
                _add_reaction_neighbor(reactdict[0], before, after)
                _add_reaction_neighbor(reactdict[1], after, before)
                if before_conflicts[changed_index]:
                    _add_reaction_neighbor(
                        reactdict[0], before, ReactionsFinder.CONFLICT
                    )
                if after_conflicts[changed_index]:
                    _add_reaction_neighbor(
                        reactdict[1], after, ReactionsFinder.CONFLICT
                    )
    counter = Counter()
    for reaction in dps_reaction(reactdict):
        if (
            ReactionsFinder.EMPTY in reaction[0]
            or ReactionsFinder.EMPTY in reaction[1]
            or ReactionsFinder.CONFLICT in reaction[0]
            or ReactionsFinder.CONFLICT in reaction[1]
        ):
            continue
        pair = _filter_reaction_pair(reaction, molecule_names)
        if pair is not None:
            counter[pair] += 1
    return counter


def _get_transition_reactions_by_index(transition_index):
    """Multiprocessing entry point receiving only a transition index."""
    assert _REACTION_ATOMEACH is not None
    return _get_transition_reaction_result(
        transition_index,
        _REACTION_ATOMEACH,
        _REACTION_CONFLICT,
        _REACTION_MNAME,
    )


def _get_transition_reaction_result(
    transition_index,
    atomeach,
    conflict,
    molecule_names,
):
    """Return one indexed transition result for parent or worker execution."""
    transition_index = int(transition_index)
    return transition_index, _calculate_transition_reactions(
        atomeach[:, transition_index],
        atomeach[:, transition_index + 1],
        conflict[:, transition_index],
        conflict[:, transition_index + 1],
        molecule_names,
    )


class ReactionsFinder(SharedRNGData):
    """Find and aggregate reactions between adjacent analyzed frames."""

    CONFLICT = -1
    EMPTY = 0

    step: int
    mname: np.ndarray | _MoleculeNameTable
    reactionabcdfilename: str
    printreactionevent: bool
    nproc: int

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            [
                "step",
                "mname",
                "reactionabcdfilename",
                "printreactionevent",
                "nproc",
            ],
            [],
        )

    def findreactions(
        self,
        matrix_store,
        timed_store=None,
        modified_atom_events=None,
        active_transitions=None,
    ):
        """Analyze transitions from shared mappings and stream compact counters."""
        if self.printreactionevent and timed_store is None:
            raise ValueError("printreactionevent requires a timed-output HDF5 store")
        assert matrix_store.mname_path is not None
        assert matrix_store.mname_values_path is not None
        reaction_counter = Counter()
        transition_count = max(0, self.step - 1)
        active_transition_index_available = active_transitions is not None
        transition_indices = _normalize_active_transitions(
            active_transitions,
            transition_count,
        )
        if transition_indices is None:
            transition_indices = range(transition_count)
            active_transition_count = transition_count
        else:
            active_transition_count = len(transition_indices)
            logger.info(
                "Reaction active transitions: %d / %d from atom routes",
                active_transition_count,
                transition_count,
            )
        if timed_store is not None:
            timed_store.reaction_total_transition_count = transition_count
            timed_store.reaction_active_transition_count = active_transition_count
            timed_store.reaction_active_transition_index_available = (
                active_transition_index_available
            )
        atom_count = int(matrix_store.shape[0])
        reaction_start_method = get_start_method()
        reaction_nproc = _reaction_worker_count(
            self.nproc,
            atom_count,
            active_transition_count,
            modified_atom_events,
            reaction_start_method,
        )
        if reaction_nproc < self.nproc:
            event_description = (
                "unknown"
                if modified_atom_events is None
                else f"{int(modified_atom_events)}"
            )
            logger.info(
                "Reaction worker count reduced from %d to %d for %s modified "
                "atom events across %d atom-transition values using %s start",
                self.nproc,
                reaction_nproc,
                event_description,
                atom_count * active_transition_count,
                reaction_start_method,
            )
        if reaction_nproc == 1:
            results = (
                _get_transition_reaction_result(
                    transition_index,
                    matrix_store.atomeach,
                    matrix_store.conflict,
                    self.mname,
                )
                for transition_index in tqdm(
                    transition_indices,
                    total=active_transition_count,
                    desc="Analyze reactions (A+B->C+D)",
                    unit="timestep",
                    disable=None,
                )
            )
        else:
            reaction_chunksize = _reaction_chunksize(
                reaction_nproc,
                atom_count,
                active_transition_count,
                modified_atom_events,
            )
            reaction_max_inflight = max(
                reaction_chunksize,
                2 * reaction_nproc * reaction_chunksize,
            )
            logger.info(
                "Reaction multiprocessing: %d workers, chunksize=%d, "
                "max_inflight=%d",
                reaction_nproc,
                reaction_chunksize,
                reaction_max_inflight,
            )
            results = run_mp(
                reaction_nproc,
                func=_get_transition_reactions_by_index,
                l=transition_indices,
                unordered=True,
                chunksize=reaction_chunksize,
                max_inflight=reaction_max_inflight,
                initializer=_initialize_reaction_worker,
                initargs=(
                    matrix_store.atomeach_path,
                    matrix_store.conflict_path,
                    matrix_store.shape,
                    matrix_store.molecule_dtype.str,
                    matrix_store.mname_path,
                    matrix_store.mname_values_path,
                ),
                maxtasksperchild=None,
                total=active_transition_count,
                desc="Analyze reactions (A+B->C+D)",
                unit="timestep",
            )
        for transition_index, events in results:
            if self.printreactionevent:
                assert timed_store is not None
                timed_store.stage_reaction_events(transition_index, events)
            else:
                reaction_counter.update(events)

        if self.printreactionevent:
            assert timed_store is not None
            timed_store.finalize_reactions()
            summaries = timed_store.iter_reaction_summaries()
        else:
            summaries = sorted(
                (
                    (count, reactant, product)
                    for (reactant, product), count in reaction_counter.items()
                ),
                key=lambda row: (-row[0], row[1], row[2]),
            )
        with WriteBuffer(open(self.reactionabcdfilename, "w"), sep="\n") as handle:
            for count, reactant, product in summaries:
                handle.append(f"{count} {reactant}->{product}")

    def _getstepreaction(self, item):
        """Compatibility helper for unit tests using serialized frame arrays."""
        item = bytestolist(item)
        if self.printreactionevent:
            transition_index = int(item[0])
            arrays = item[1:]
        else:
            transition_index = None
            arrays = item
        counter = _calculate_transition_reactions(*arrays, self.mname)
        if self.printreactionevent:
            assert transition_index is not None
            return transition_index, counter
        return [
            f"{reactant}->{product}"
            for (reactant, product), count in counter.items()
            for _ in range(count)
        ]

    def _filterreactionpair(self, reaction):
        return _filter_reaction_pair(reaction, self.mname)

    def _filterspec(self, reaction):
        reactionpair = self._filterreactionpair(reaction)
        if reactionpair is None:
            return None
        return "->".join(reactionpair)
