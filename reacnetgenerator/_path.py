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
from collections import Counter, defaultdict

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from ._logging import logger
from ._reaction import ReactionsFinder
from ._timedoutput import TimedOutputStore
from .utils import (
    SharedRNGData,
    WriteBuffer,
    bytestolist,
    get_timestep_value,
    listtostirng,
    read_compressed_block,
    run_mp,
)

_MATRIX_WRITE_CELLS = 1024 * 1024
_MATRIX_WRITE_ATOMS = 65536
_ROUTE_ATOMEACH = None
_ROUTE_ATOMTYPE = None
_ROUTE_ATOMNAME = None
_ROUTE_SELECTATOMS = None
_ROUTE_MNAME = None
_ROUTE_FRAME_RANGE = None


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
        self.conflict = np.memmap(
            self.conflict_path,
            mode="w+",
            dtype=np.bool_,
            shape=self.shape,
        )
        self.mname_path = None
        self.atomtype_path = None

    def save_molecule_names(self, values):
        """Save names once so workers map them instead of receiving copies."""
        handle, self.mname_path = tempfile.mkstemp(
            prefix="reacnetgenerator-mname-", suffix=".npy"
        )
        os.close(handle)
        np.save(self.mname_path, np.asarray(values), allow_pickle=False)

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

    def close(self):
        """Close mappings and remove their temporary backing files."""
        for name in ("atomeach", "conflict"):
            value = getattr(self, name, None)
            if value is not None:
                value.flush()
                mmap = getattr(value, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                setattr(self, name, None)
        for path in (
            self.atomeach_path,
            self.conflict_path,
            self.mname_path,
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
    atomtype_path,
    atomname,
    selectatoms,
    frame_range,
):
    """Attach one route worker to read-only shared mappings."""
    global _ROUTE_ATOMEACH
    global _ROUTE_ATOMTYPE
    global _ROUTE_ATOMNAME
    global _ROUTE_SELECTATOMS
    global _ROUTE_MNAME
    global _ROUTE_FRAME_RANGE
    _ROUTE_ATOMEACH = np.memmap(
        atomeach_path,
        mode="r",
        dtype=np.dtype(dtype_string),
        shape=tuple(shape),
    )
    _ROUTE_ATOMTYPE = np.load(atomtype_path, mmap_mode="r", allow_pickle=False)
    _ROUTE_ATOMNAME = np.asarray(atomname)
    _ROUTE_SELECTATOMS = set(selectatoms)
    _ROUTE_MNAME = np.load(mname_path, mmap_mode="r", allow_pickle=False)
    _ROUTE_FRAME_RANGE = frame_range


def _calculate_atom_route(
    atom_index,
    timeline,
    atomtype,
    atomname,
    selectatoms,
    molecule_names,
):
    """Calculate one atom route without owning the full atom-frame matrix."""
    timeline = timeline[np.flatnonzero(timeline)]
    if timeline.size:
        change_time = np.concatenate(
            [
                np.zeros((1,), dtype=int),
                np.flatnonzero(np.diff(timeline)) + 1,
            ]
        )
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
    return molecule_route, route_string


def _get_atom_route_by_index(atom_index):
    """Multiprocessing entry point receiving only an integer atom index."""
    assert _ROUTE_ATOMEACH is not None
    assert _ROUTE_FRAME_RANGE is not None
    start, stop = _ROUTE_FRAME_RANGE
    return _calculate_atom_route(
        int(atom_index),
        _ROUTE_ATOMEACH[int(atom_index), start:stop],
        _ROUTE_ATOMTYPE,
        _ROUTE_ATOMNAME,
        _ROUTE_SELECTATOMS,
        _ROUTE_MNAME,
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
    mname: np.ndarray

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
            ],
            ["mname", "atomnames", "allmoleculeroute", "splitmoleculeroute"],
        )
        self._moleculeframefilter = self._getmoleculefilterset(self.moleculeframes)
        self._moleculetimestepfilter = self._getmoleculefilterset(
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
        self._printmoleculename(timed_store)
        matrix_store = self._getatomeach()
        matrix_store.save_molecule_names(self.mname)
        matrix_store.save_atom_types(self.atomtype)
        try:
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
            ReactionsFinder(self.rng).findreactions(
                matrix_store,
                timed_store=timed_store,
            )
        finally:
            matrix_store.close()

    @abstractmethod
    def _printmoleculename(self, timed_store):
        pass

    def _getatomeach(self):
        """Build disk-backed atom-frame matrices; molecule IDs start from 1."""
        molecule_dtype = self._molecule_index_dtype(self.hmmit)
        store = _AtomFrameStore((self.N, self.step), molecule_dtype)
        try:
            with (
                open(
                    self.hmmfilename if self.runHMM else self.originfilename, "rb"
                ) as fh,
                open(self.moleculetemp2filename, "rb") as ft,
            ):
                for molecule_id, (linehz, linetz) in enumerate(
                    tqdm(
                        zip(
                            read_compressed_block(fh),
                            itertools.zip_longest(*[read_compressed_block(ft)] * 4),
                        ),
                        total=self.hmmit,
                        desc="Analyze atoms",
                        unit="molecule",
                        disable=None,
                    ),
                    start=1,
                ):
                    signal = bytestolist(linehz)
                    atoms = np.asarray(bytestolist(linetz[0]), dtype=np.int64)
                    frames = np.flatnonzero(signal)
                    for atom_start in range(0, len(atoms), _MATRIX_WRITE_ATOMS):
                        atom_batch = atoms[
                            atom_start : atom_start + _MATRIX_WRITE_ATOMS
                        ]
                        frames_per_batch = max(
                            1, _MATRIX_WRITE_CELLS // max(1, len(atom_batch))
                        )
                        for frame_start in range(0, len(frames), frames_per_batch):
                            frame_batch = frames[
                                frame_start : frame_start + frames_per_batch
                            ]
                            selected = store.atomeach[np.ix_(atom_batch, frame_batch)]
                            overlap_atoms, overlap_frames = np.nonzero(selected)
                            if overlap_atoms.size:
                                store.conflict[
                                    atom_batch[overlap_atoms],
                                    frame_batch[overlap_frames],
                                ] = True
                            store.atomeach[np.ix_(atom_batch, frame_batch)] = (
                                molecule_id
                            )
            store.flush()
            return store
        except BaseException:
            store.close()
            raise

    @staticmethod
    def _molecule_index_dtype(hmmit):
        """Return the smallest unsigned dtype that can store every molecule ID."""
        maximum = max(1, int(hmmit))
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            if maximum <= np.iinfo(dtype).max:
                return np.dtype(dtype)
        raise OverflowError(
            "Too many molecules to index with an unsigned 64-bit integer"
        )

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
        assert matrix_store.mname_path is not None
        assert matrix_store.atomtype_path is not None
        with WriteBuffer(
            open(
                (
                    self.atomroutefilename
                    if timeaxis is None
                    else f"{self.atomroutefilename}.{timeaxis}"
                ),
                "w",
            ),
            sep="\n",
        ) as f:
            allmoleculeroute = set() if self.runHMM else []
            if not self.runHMM:
                have_added = {}
            else:
                have_added = None
            results = run_mp(
                self.nproc,
                func=_get_atom_route_by_index,
                l=range(self.N),
                unordered=False,
                chunksize=1,
                max_inflight=max(2, 2 * self.nproc),
                initializer=_initialize_route_worker,
                initargs=(
                    matrix_store.atomeach_path,
                    matrix_store.shape,
                    matrix_store.molecule_dtype.str,
                    matrix_store.mname_path,
                    matrix_store.atomtype_path,
                    self.atomname,
                    tuple(self.selectatoms),
                    tuple(frame_range),
                ),
                maxtasksperchild=None,
                total=self.N,
                desc=(
                    "Collect reaction paths"
                    if timeaxis is None
                    else f"Collect reaction paths {timeaxis}"
                ),
                unit="atom",
            )
            for ii, (moleculeroute, routestr) in enumerate(results):
                f.append(routestr)
                if moleculeroute.size > 0:
                    if not self.runHMM:
                        # check whether repeated or not if analyzing without HMM
                        for rr in moleculeroute:
                            tpr = tuple(rr)
                            assert have_added is not None
                            if have_added.get(tpr, matrix_store.shape[0]) >= ii:
                                have_added[tpr] = ii
                                allmoleculeroute.append(rr.reshape(1, 2))
                    else:
                        assert isinstance(allmoleculeroute, set)
                        allmoleculeroute.update(
                            (int(pair[0]), int(pair[1])) for pair in moleculeroute
                        )
        if self.runHMM:
            return (
                np.asarray(sorted(allmoleculeroute), dtype=int).reshape((-1, 2))
                if allmoleculeroute
                else np.zeros((0, 2), dtype=int)
            )
        assert isinstance(allmoleculeroute, list)
        return (
            np.concatenate(allmoleculeroute)
            if allmoleculeroute
            else np.zeros((0, 2), dtype=int)
        )

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
        Satom = sorted(self.atomname, key=lambda x: len(x), reverse=True)
        elements = "|".join(
            [
                ((an.upper() + "|" + an.lower()) if len(an) == 1 else an)
                for an in Satom
                if an != "H"
            ]
        )
        smi = re.sub(r"(?<!\[)(" + elements + r")(?!H|\])", r"[\1]", smi)
        return smi.replace("[HH]", "[H]")

    def convertSMILES(self, atoms, bonds):
        """Convert atoms and bonds information to SMILES.

        Raises
        ------
        ValueError
            (RDKit error) Maximum BFS search size exceeded.
        """
        m = Chem.RWMol(Chem.MolFromSmiles(""))
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
        atoms = np.array(bytestolist(line[0]), dtype=int)
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
    def _getmoleculefilterset(cls, values):
        return set(values) if cls._hasmoleculefilter(values) else None

    def _shouldprintmoleculetimelinerow(self, frame, timestep):
        if (
            self._moleculeframefilter is not None
            and int(frame) not in self._moleculeframefilter
        ):
            return False
        return not (
            self._moleculetimestepfilter is not None
            and int(timestep) not in self._moleculetimestepfilter
        )

    def _formatmoleculename(self, name, atoms, bonds):
        return listtostirng((name, atoms, bonds), sep=(" ", ";", ","))

    @staticmethod
    def _formatmoleculeatomids(atoms):
        return ";".join(str(atom) for atom in atoms)

    @staticmethod
    def _formatmoleculebondids(bonds):
        return ";".join("-".join(str(item) for item in bond) for bond in bonds)

    def _itermoleculeranges(self, frames):
        selected_frames = np.asarray(frames)
        start = previous = None
        for frame_value in selected_frames:
            frame = int(frame_value)
            if not self._shouldprintmoleculetimelinerow(
                frame,
                int(get_timestep_value(self.timestep[frame])),
            ):
                continue
            if start is None:
                start = previous = frame
                continue
            assert previous is not None
            if frame == previous:
                continue
            if frame != previous + 1:
                yield start, previous
                start = frame
            previous = frame
        if start is not None:
            assert previous is not None
            yield start, previous

    def _getmoleculeranges(self, frames):
        return list(self._itermoleculeranges(frames))

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


class _CollectMolPaths(_CollectPaths):
    """VF2 is used to identify isomers.

    If SMILES is failed to generate, fallback to the name like CxHyOz.
    """

    def _printmoleculename(self, timed_store):
        mname = []
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
        self.mname = np.array(mname)


class _CollectSMILESPaths(_CollectPaths):
    def _printmoleculename(self, timed_store):
        mname = []
        d = defaultdict(list)
        name_mapping = {}
        name_mapping_graph = defaultdict(dict)
        em = iso.numerical_edge_match(["atom", "level"], ["None", 1])
        self.n_unknown = 0
        with (
            WriteBuffer(open(self.moleculefilename, "w"), sep="\n") as fm,
            open(self.moleculetemp2filename, "rb") as ft,
        ):
            results = run_mp(
                self.nproc,
                func=self._calmoleculeSMILESname,
                l=read_compressed_block(ft),
                unordered=False,
                chunksize=1,
                max_inflight=max(2, 2 * self.nproc),
                nlines=4,
                total=self.hmmit,
                desc="Indentify isomers",
                unit="molecule",
            )
            for molecule_id, (
                name,
                atoms,
                bonds,
                frame_block,
            ) in enumerate(results, start=1):
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
                if frame_block is not None:
                    self._storetimedmolecule(
                        timed_store,
                        molecule_id,
                        name,
                        atoms,
                        bonds,
                        bytestolist(frame_block),
                    )
        self.mname = np.array(mname)

    def _calmoleculeSMILESname(self, item):
        line = item
        atoms, bonds = self._getatomsandbonds(line)
        try:
            name = self.convertSMILES(atoms, bonds)
        except ValueError:
            # fallback to VF2
            name = None
        frame_block = line[-1] if self._needmoleculetimeline() else None
        return name, atoms, bonds, frame_block


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
