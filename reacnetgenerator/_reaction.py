# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Reactions finder."""

from collections import Counter, defaultdict

import numpy as np

from .dps import dps_reaction  # type:ignore
from .utils import SharedRNGData, WriteBuffer, bytestolist, run_mp

_REACTION_ATOMEACH = None
_REACTION_CONFLICT = None
_REACTION_MNAME = None


def _initialize_reaction_worker(
    atomeach_path,
    conflict_path,
    shape,
    molecule_dtype_string,
    mname_path,
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
    _REACTION_CONFLICT = np.memmap(
        conflict_path,
        mode="r",
        dtype=np.bool_,
        shape=tuple(shape),
    )
    _REACTION_MNAME = np.load(mname_path, mmap_mode="r", allow_pickle=False)


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
    modified_atoms = np.flatnonzero(atomeach_before != atomeach_after)
    reactdict = [defaultdict(list), defaultdict(list)]
    for atom_index in modified_atoms:
        before = int(atomeach_before[atom_index])
        after = int(atomeach_after[atom_index])
        reactdict[0][before].append(after)
        reactdict[1][after].append(before)
        if conflict_before[atom_index]:
            reactdict[0][before].append(ReactionsFinder.CONFLICT)
        if conflict_after[atom_index]:
            reactdict[1][after].append(ReactionsFinder.CONFLICT)
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
    transition_index = int(transition_index)
    return transition_index, _calculate_transition_reactions(
        _REACTION_ATOMEACH[:, transition_index],
        _REACTION_ATOMEACH[:, transition_index + 1],
        _REACTION_CONFLICT[:, transition_index],
        _REACTION_CONFLICT[:, transition_index + 1],
        _REACTION_MNAME,
    )


class ReactionsFinder(SharedRNGData):
    """Find and aggregate reactions between adjacent analyzed frames."""

    CONFLICT = -1
    EMPTY = 0

    step: int
    mname: np.ndarray
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

    def findreactions(self, matrix_store, timed_store=None):
        """Analyze transitions from shared mappings and stream compact counters."""
        if self.printreactionevent and timed_store is None:
            raise ValueError("printreactionevent requires a timed-output HDF5 store")
        assert matrix_store.mname_path is not None
        reaction_counter = Counter()
        results = run_mp(
            self.nproc,
            func=_get_transition_reactions_by_index,
            l=range(max(0, self.step - 1)),
            unordered=True,
            chunksize=1,
            max_inflight=max(2, 2 * self.nproc),
            initializer=_initialize_reaction_worker,
            initargs=(
                matrix_store.atomeach_path,
                matrix_store.conflict_path,
                matrix_store.shape,
                matrix_store.molecule_dtype.str,
                matrix_store.mname_path,
            ),
            maxtasksperchild=None,
            total=max(0, self.step - 1),
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
            summaries = (
                (count, reactant, product)
                for (reactant, product), count in reaction_counter.most_common()
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
