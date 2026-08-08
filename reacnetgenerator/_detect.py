# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Detect molecules.

There are two types of input files that could be imported by ReacNetGenerator,
the first part of necessary is the trajectory from reactive MD, the second
part can be the bond information normally given by simulation using ReaxFF.
In fact, atomic coordinates can be converted to the bond information with
the Open Babel software.[1]_ As a results, ReacNetGenerator can both processes ReaxFF
trajectories, AIMD trajectories, and other kinds of reactive trajectories.
With the bond information, molecules can be detected from atoms by Depth-first
search at every timestep.[2]_ By using this way, all molecules in a given
trajectory can be acquired. Molecules consisting of same atoms and bonds can
be considered as the same molecule.

References
----------
.. [1] O'Boyle, N. M.; Banck, M.; James, C. A.; Morley, C.; Vandermeersch, T.;
   Hutchison, G. Open Babel: An open chemical toolbox. J. Cheminf. 2011, 3(1),
   33-47.
.. [2] Tarjan, R. Depth-first search and linear graph algorithms. SIAM J. Comput.
   1972, 1 (2), 146-160.
"""

import itertools
import operator
import os
import tempfile
from abc import ABCMeta, abstractmethod
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from enum import Enum, IntEnum, auto
from multiprocessing import get_start_method
from time import perf_counter
from typing import ClassVar

import numpy as np
from tqdm.auto import tqdm

try:
    from openbabel import __version__ as obversion
    from openbabel import openbabel
except ImportError:  # pragma: no cover
    raise ImportError("Open Babel 3.1.0 is required.")
from ase import Atom, Atoms
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.neighborlist import natural_cutoffs, neighbor_list
from packaging import version
from scipy.spatial import cKDTree

from ._logging import logger
from .dps import dps  # type:ignore
from .utils import SharedRNGData, WriteBuffer, listtobytes, run_mp

if version.parse(obversion) < version.parse("3.1.0"):  # pragma: no cover
    raise ImportError("Open Babel 3.1.0 is required.")

openbabel.obErrorLog.StopLogging()

_DETECT_NONFORK_INPUT_BYTES_PER_WORKER = 8 * 1024 * 1024
_DETECT_THREAD_COMPRESSION_MIN_PAYLOAD_BYTES = 16 * 1024 * 1024
_DETECT_THREAD_COMPRESSION_MIN_AVERAGE_BYTES = 128 * 1024
_DETECT_COMPRESSION_THREAD_WORKERS = 2
_FRAME_INDEX_TYPECODES = ("B", "H", "I", "Q")
_OPENBABEL_PERIODIC_NEIGHBOR_MIN_ATOMS = 2048
_OPENBABEL_COVALENT_RADIUS_TOLERANCE = 0.45
_OPENBABEL_MIN_BOND_DISTANCE = 0.4
_OPENBABEL_MIN_BOND_ANGLE = 45.0
_OPENBABEL_DISTANCE_AMBIGUITY = 1e-7


class _BondDetectionMode(IntEnum):
    """Compact worker result with its production log label."""

    def __new__(cls, value: int, label: str):
        member = int.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    UNKNOWN = 0, "unknown"
    CKDTREE = 1, "periodic-cKDTree"
    ASE = 2, "ASE"
    OPENBABEL_NONPERIODIC = 3, "OpenBabel-nonperiodic"
    OPENBABEL_SMALL_FRAME = 4, "OpenBabel-below-threshold"
    OPENBABEL_UNSUPPORTED_CELL = 5, "OpenBabel-unsupported-cell"
    OPENBABEL_NARROW_CELL = 6, "OpenBabel-narrow-cell"
    OPENBABEL_INVALID_COORDINATES = 7, "OpenBabel-invalid-coordinates"
    OPENBABEL_INVALID_RADII = 8, "OpenBabel-invalid-radii"
    OPENBABEL_CUTOFF_BOUNDARY = 9, "OpenBabel-cutoff-boundary"


# Compatibility names keep the existing internal tests and monkeypatches stable.
_BOND_DETECTION_UNKNOWN = _BondDetectionMode.UNKNOWN
_BOND_DETECTION_CKDTREE = _BondDetectionMode.CKDTREE
_BOND_DETECTION_ASE = _BondDetectionMode.ASE
_BOND_DETECTION_OPENBABEL_NONPERIODIC = _BondDetectionMode.OPENBABEL_NONPERIODIC
_BOND_DETECTION_OPENBABEL_SMALL_FRAME = _BondDetectionMode.OPENBABEL_SMALL_FRAME
_BOND_DETECTION_OPENBABEL_UNSUPPORTED_CELL = (
    _BondDetectionMode.OPENBABEL_UNSUPPORTED_CELL
)
_BOND_DETECTION_OPENBABEL_NARROW_CELL = _BondDetectionMode.OPENBABEL_NARROW_CELL
_BOND_DETECTION_OPENBABEL_INVALID_COORDINATES = (
    _BondDetectionMode.OPENBABEL_INVALID_COORDINATES
)
_BOND_DETECTION_OPENBABEL_INVALID_RADII = _BondDetectionMode.OPENBABEL_INVALID_RADII
_BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY = _BondDetectionMode.OPENBABEL_CUTOFF_BOUNDARY


def _widen_frame_indices(values, frame):
    """Append an overflowing frame index using the next viable width."""
    try:
        start = _FRAME_INDEX_TYPECODES.index(values.typecode) + 1
    except ValueError:
        raise TypeError(
            f"Unsupported frame index typecode {values.typecode!r}"
        ) from None
    last_error = None
    for typecode in _FRAME_INDEX_TYPECODES[start:]:
        if array(typecode).itemsize <= values.itemsize:
            continue
        widened = array(typecode, values)
        try:
            widened.append(frame)
        except OverflowError as error:
            last_error = error
            continue
        return widened
    if last_error is not None:
        raise last_error
    raise OverflowError(
        f"Frame index {frame!r} exceeds available unsigned array storage"
    )


class _FrameSourceMap(Mapping[int, tuple[int, int]]):
    """Store sequential frame provenance as arithmetic source segments."""

    __slots__ = (
        "_last_source_id",
        "_length",
        "_next_source_frame",
        "_source_frames",
        "_source_ids",
        "_starts",
        "_stepinterval",
    )

    def __init__(self, stepinterval):
        self._stepinterval = max(1, int(stepinterval))
        self._length = 0
        self._last_source_id = None
        self._next_source_frame = None
        self._starts = array("Q")
        self._source_ids = array("I")
        self._source_frames = array("Q")

    def append(self, step, source_id, source_frame):
        """Append one analyzed frame in input order."""
        step = operator.index(step)
        source_id = operator.index(source_id)
        source_frame = operator.index(source_frame)
        if step != self._length:
            raise RuntimeError(
                "Detect frame results must be consumed in sequential input order"
            )
        if source_id <= 0 or source_frame < 0:
            raise ValueError("Frame source metadata must be non-negative")
        start_new_segment = (
            source_id != self._last_source_id or source_frame != self._next_source_frame
        )
        if start_new_segment:
            self._starts.append(step)
            self._source_ids.append(source_id)
            self._source_frames.append(source_frame)
        self._last_source_id = source_id
        self._next_source_frame = source_frame + self._stepinterval
        self._length += 1

    @property
    def nbytes(self):
        """Return bytes used by the compact segment payloads."""
        return sum(
            values.itemsize * len(values)
            for values in (self._starts, self._source_ids, self._source_frames)
        )

    @property
    def segment_count(self):
        """Return the number of arithmetic provenance segments."""
        return len(self._starts)

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._length))

    def __getitem__(self, frame):
        try:
            frame = operator.index(frame)
        except TypeError:
            raise KeyError(frame) from None
        if frame < 0 or frame >= self._length:
            raise KeyError(frame)
        segment = bisect_right(self._starts, frame) - 1
        source_frame = (
            self._source_frames[segment]
            + (frame - self._starts[segment]) * self._stepinterval
        )
        return int(self._source_ids[segment]), int(source_frame)

    def fill_arrays(self, start, stop, source_ids, source_frames):
        """Fill one contiguous provenance block without per-frame lookups."""
        start = operator.index(start)
        stop = operator.index(stop)
        if start < 0 or stop < start or stop > self._length:
            raise IndexError("Frame provenance block is outside the analyzed range")
        if len(source_ids) != stop - start or len(source_frames) != stop - start:
            raise ValueError("Frame provenance output arrays have the wrong length")
        if start == stop:
            return
        segment = bisect_right(self._starts, start) - 1
        frame = start
        while frame < stop:
            next_start = (
                self._starts[segment + 1]
                if segment + 1 < len(self._starts)
                else self._length
            )
            segment_stop = min(stop, next_start)
            output_start = frame - start
            output_stop = segment_stop - start
            source_ids[output_start:output_stop] = self._source_ids[segment]
            relative_start = frame - self._starts[segment]
            relative_stop = segment_stop - self._starts[segment]
            source_frames[output_start:output_stop] = self._source_frames[
                segment
            ] + np.arange(relative_start, relative_stop, dtype=np.uint64) * int(
                self._stepinterval
            )
            frame = segment_stop
            segment += 1


def _detect_worker_count(
    requested_nproc,
    input_bytes,
    start_method=None,
):
    """Amortize non-fork imports while retaining cheap fork parallelism."""
    maximum_workers = max(1, int(requested_nproc))
    if start_method is None:
        start_method = get_start_method()
    if start_method == "fork":
        return maximum_workers
    input_bytes = max(0, int(input_bytes))
    amortized_workers = max(
        1,
        (input_bytes + _DETECT_NONFORK_INPUT_BYTES_PER_WORKER - 1)
        // _DETECT_NONFORK_INPUT_BYTES_PER_WORKER,
    )
    return min(maximum_workers, amortized_workers)


def _detect_compression_worker_count(
    requested_nproc,
    molecule_count,
    payload_bytes,
):
    """Select shared-memory compression threads only when work amortizes them."""
    maximum_workers = max(1, int(requested_nproc))
    molecule_count = max(0, int(molecule_count))
    payload_bytes = max(0, int(payload_bytes))
    if maximum_workers < 2 or molecule_count < 2:
        return 1
    if payload_bytes < _DETECT_THREAD_COMPRESSION_MIN_PAYLOAD_BYTES:
        return 1
    if payload_bytes < molecule_count * _DETECT_THREAD_COMPRESSION_MIN_AVERAGE_BYTES:
        return 1
    return min(maximum_workers, _DETECT_COMPRESSION_THREAD_WORKERS)


def _bounded_thread_map(func, values, *, workers, max_inflight):
    """Yield ordered thread results while retaining only bounded futures."""
    workers = int(workers)
    max_inflight = int(max_inflight)
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if max_inflight <= 0:
        raise ValueError("max_inflight must be a positive integer")
    values = iter(values)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        try:
            for _ in range(max_inflight):
                try:
                    value = next(values)
                except StopIteration:
                    break
                pending.append(executor.submit(func, value))
            while pending:
                yield pending.popleft().result()
                try:
                    value = next(values)
                except StopIteration:
                    continue
                pending.append(executor.submit(func, value))
        finally:
            for future in pending:
                future.cancel()


class _Detect(SharedRNGData, metaclass=ABCMeta):
    """Detect molecules.

    Parameters
    ----------
    rng: reacnetgenerator.ReacNetGenerator
        The ReacNetGenerator class.
    """

    subclasses: ClassVar[dict] = {}

    # type hints
    # TODO: we need a better way to communicate with parameters
    inputfilename: list
    atomname: np.ndarray
    stepinterval: int
    nproc: int
    pbc: bool
    cell: np.ndarray

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            ["inputfilename", "atomname", "stepinterval", "nproc", "pbc", "cell"],
            [
                "N",
                "atomtype",
                "step",
                "timestep",
                "framesource",
                "temp1it",
                "moleculetempfilename",
            ],
        )

    @classmethod
    def gettype(cls, rng):
        """Get the class for the input file type.

        Now ReacNetGen support the following files:
            - lammpsbondfile: LAMMPS bond files, see http://lammps.sandia.gov/doc/fix_reax_bonds.html
            - lammpsdumpfile: LAMMPS dump files, see https://lammps.sandia.gov/doc/dump.html
            - xyz: xyz files
            - extxyz: extended xyz files

        Parameters
        ----------
        rng : reacnetgenerator.ReacNetGenerator
            The ReacNetGenerator class.

        Returns
        -------
        cls.subclasses[rng.inputfiletype](rng): class
            The _Detect class for specific file type.
        """
        if rng.inputfiletype not in cls.subclasses:
            raise ValueError(f"Unsupported input file type {rng.inputfiletype}")
        return cls.subclasses[rng.inputfiletype](rng)

    @classmethod
    def register_subclass(cls, message_type):
        """Register the file type, used as a decorator.

        Parameters
        ----------
        message_type : str
            The type name to register, such as `xyz`.

        Returns
        -------
        decorator: function
            The decorator that used for a subclass.

        Examples
        --------
        @_Detect.register_subclass("lammpsbondfile")
        class _DetectLAMMPSbond(_Detect):
        """

        def decorator(subclass):
            cls.subclasses[message_type] = subclass
            return subclass

        return decorator

    def detect(self):
        """Detect molecules."""
        self._readinputfile()
        self.returnkeys()

    def _readinputfile(self):
        """Read the input file."""
        d = defaultdict(lambda: array("H"))
        timestep = array("q")
        framesource = _FrameSourceMap(self.stepinterval)
        start_method = get_start_method()
        input_bytes = sum(os.path.getsize(filename) for filename in self.inputfilename)
        detect_nproc = _detect_worker_count(
            self.nproc,
            input_bytes,
            start_method,
        )
        if detect_nproc < self.nproc:
            logger.info(
                "Detect worker count reduced from %d to %d for %.3f MiB "
                "of input using %s start",
                self.nproc,
                detect_nproc,
                input_bytes / (1024 * 1024),
                start_method,
            )
        logger.info(
            "Detect frame execution: %d worker(s), chunksize=1, max_inflight=%d, "
            "start=%s",
            detect_nproc,
            max(2, 2 * detect_nproc),
            start_method,
        )
        with open(self.inputfilename[0]) as f:
            _steplinenum = self._readNfunc(f)
        results = run_mp(
            detect_nproc,
            func=self._readstepwithsource,
            l=self._iterinputsteps(_steplinenum),
            unordered=False,
            chunksize=1,
            max_inflight=max(2, 2 * detect_nproc),
            desc="Read bond information and Detect molecules",
            unit="timestep",
        )
        bond_detection_counts = Counter()
        for molecules, (
            step,
            source_id,
            source_frame,
            thetimestep,
            bond_detection_mode,
        ) in results:
            for molecule in molecules:
                frame_indices = d[molecule]
                try:
                    frame_indices.append(step)
                except OverflowError:
                    d[molecule] = _widen_frame_indices(frame_indices, step)
            if step != len(timestep):
                raise RuntimeError(
                    "Detect frame results must be consumed in sequential input order"
                )
            timestep.append(thetimestep)
            framesource.append(step, source_id, source_frame)
            if bond_detection_mode != _BOND_DETECTION_UNKNOWN:
                if not bond_detection_counts:
                    logger.info(
                        "Coordinate bond perception first frame: %s",
                        _BondDetectionMode(bond_detection_mode).label,
                    )
                bond_detection_counts[bond_detection_mode] += 1
        if bond_detection_counts:
            logger.info(
                "Coordinate bond perception: %d frame(s); %s",
                sum(bond_detection_counts.values()),
                ", ".join(
                    f"{mode.label}={bond_detection_counts[mode]}"
                    for mode in _BondDetectionMode
                    if mode is not _BondDetectionMode.UNKNOWN
                    if bond_detection_counts[mode]
                ),
            )
        self.temp1it = len(d)
        frame_type_counts = Counter()
        frame_value_count = 0
        frame_payload_bytes = 0
        for values in d.values():
            value_count = len(values)
            frame_type_counts[values.typecode] += 1
            frame_value_count += value_count
            frame_payload_bytes += value_count * values.itemsize
        logger.info(
            "Detect molecule-frame storage: %d occurrences in %.3f MiB across "
            "%d molecules (%s)",
            frame_value_count,
            frame_payload_bytes / (1024 * 1024),
            self.temp1it,
            ", ".join(
                f"{typecode}:{frame_type_counts[typecode]}"
                for typecode in _FRAME_INDEX_TYPECODES
                if frame_type_counts[typecode]
            ),
        )
        average_frame_payload_bytes = frame_payload_bytes // max(1, self.temp1it)
        compression_workers = _detect_compression_worker_count(
            self.nproc,
            self.temp1it,
            frame_payload_bytes,
        )
        if compression_workers == 1:
            logger.info(
                "Molecule-frame compression: parent process for %.3f MiB "
                "compact payload (%.3f KiB/molecule); array IPC disabled",
                frame_payload_bytes / (1024 * 1024),
                average_frame_payload_bytes / 1024,
            )
            compression_values = run_mp(
                1,
                func=self._compressvalue,
                l=d.values(),
                unordered=False,
                desc="Save molecules",
                unit="molecule",
                total=self.temp1it,
            )
        else:
            compression_max_inflight = 2 * compression_workers
            logger.info(
                "Molecule-frame compression: %d bounded parent threads for "
                "%.3f MiB compact payload (%.3f KiB/molecule, "
                "max_inflight=%d); process array IPC disabled",
                compression_workers,
                frame_payload_bytes / (1024 * 1024),
                average_frame_payload_bytes / 1024,
                compression_max_inflight,
            )
            compression_values = _bounded_thread_map(
                self._compressvalue,
                d.values(),
                workers=compression_workers,
                max_inflight=compression_max_inflight,
            )
        compression_started = perf_counter()
        with ExitStack() as compression_stack:
            compression_values = compression_stack.enter_context(
                closing(compression_values)
            )
            if compression_workers > 1:
                values_c = compression_stack.enter_context(
                    tqdm(
                        compression_values,
                        desc="Save molecules",
                        unit="molecule",
                        total=self.temp1it,
                        disable=None,
                    )
                )
            else:
                values_c = compression_values
            self._writemoleculetempfile(
                itertools.zip_longest(d.keys(), values_c, fillvalue=None)
            )
        logger.info(
            "Molecule-frame compression/write completed in %.3f s",
            perf_counter() - compression_started,
        )
        self.timestep = timestep
        self.framesource = framesource
        self.step = len(timestep)
        metadata_bytes = timestep.itemsize * len(timestep) + framesource.nbytes
        logger.info(
            "Detect frame metadata: %d frames in %.3f MiB, %d provenance segment(s)",
            self.step,
            metadata_bytes / (1024 * 1024),
            framesource.segment_count,
        )

    def _iterinputsteps(self, steplinenum):
        """Yield analyzed steps with stable source provenance."""
        raw_step = 0
        analyzed_step = 0
        for source_id, filename in enumerate(self.inputfilename, start=1):
            with open(filename) as handle:
                chunks = itertools.zip_longest(*[handle] * steplinenum)
                for source_frame, lines in enumerate(chunks):
                    if raw_step % self.stepinterval == 0:
                        yield analyzed_step, source_id, source_frame, lines
                        analyzed_step += 1
                    raw_step += 1

    def _readstepwithsource(self, item):
        step, source_id, source_frame, lines = item
        self._last_bond_detection_mode = _BOND_DETECTION_UNKNOWN
        molecules, step_timestep = self._readstepfunc((step, lines))
        return molecules, (
            step,
            source_id,
            source_frame,
            int(step_timestep[-1]),
            self._last_bond_detection_mode,
        )

    def _compressvalue(self, x):
        return listtobytes(np.asarray(x))

    @abstractmethod
    def _readNfunc(self, f) -> int:
        pass

    @abstractmethod
    def _readstepfunc(self, item) -> tuple[list[bytes], tuple[int, int]]:
        pass

    def _connectmolecule(self, bond, level):
        mols, bondlists = dps(bond, level)
        return [
            b"".join(
                (
                    listtobytes(mol),
                    listtobytes([(int(i[0]), int(i[1])) for i in bondlist]),
                    listtobytes([int(i[2]) for i in bondlist]),
                )
            )
            for mol, bondlist in zip(mols, bondlists)
        ]

    def _writemoleculetempfile(self, records):
        with WriteBuffer(tempfile.NamedTemporaryFile("wb", delete=False)) as f:
            self.moleculetempfilename = f.name
            for molecule, frames in records:
                if molecule is None or frames is None:
                    raise RuntimeError("Molecule compression result count mismatch")
                f.extend((molecule, frames))


@_Detect.register_subclass("bond")
@_Detect.register_subclass("lammpsbondfile")
class _DetectLAMMPSbond(_Detect):
    def _readNfunc(self, f) -> int:
        iscompleted = False
        N = None
        atomtype = None
        stepaindex = None
        index = -1
        for index, line in enumerate(f):
            if line[0] == "#":
                if line.startswith("# Number of particles"):
                    if iscompleted:
                        assert stepaindex is not None
                        steplinenum = index - stepaindex
                        break
                    else:
                        iscompleted = True
                        stepaindex = index
                    N = next(int(s) for s in line.split() if s.isdigit())
                    atomtype = np.zeros(N, dtype=int)
            else:
                s = line.split()
                assert atomtype is not None
                atomtype[int(s[0]) - 1] = int(s[1]) - 1
        else:
            steplinenum = index + 1
        assert N is not None and atomtype is not None
        self.N = N
        self.atomtype = atomtype
        return steplinenum

    def _readstepfunc(self, item) -> tuple[list[bytes], tuple[int, int]]:
        step, lines = item
        bond: list[tuple[int, ...] | None] = [None] * self.N
        level: list[tuple[int, ...] | None] = [None] * self.N
        timestep = None
        for line in lines:
            if line:
                if line[0] == "#":
                    if line.startswith("# Timestep"):
                        timestep = int(line.split()[-1])
                else:
                    s = line.split()
                    s0 = int(s[0]) - 1
                    s2 = int(s[2])
                    bond[s0] = tuple(map(self._get_idx, s[3 : 3 + s2]))
                    level[s0] = tuple(map(self._get_bo, s[4 + s2 : 4 + 2 * s2]))
        molecules = self._connectmolecule(bond, level)
        assert timestep is not None
        return molecules, (step, timestep)

    @staticmethod
    def _get_idx(x):
        return int(x) - 1

    @staticmethod
    def _get_bo(x):
        return max(1, round(float(x)))


class _DetectCrd(_Detect):
    use_ase: bool
    ase_cutoff_mult: float
    custom_cutoffs: str | None

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            [
                "inputfilename",
                "atomname",
                "stepinterval",
                "nproc",
                "pbc",
                "cell",
                "use_ase",
                "ase_cutoff_mult",
                "custom_cutoffs",
            ],
            [
                "N",
                "atomtype",
                "step",
                "timestep",
                "framesource",
                "temp1it",
                "moleculetempfilename",
            ],
        )
        self._parsed_custom_cutoffs = self._parse_custom_cutoffs(self.custom_cutoffs)
        self._atomic_numbers_by_type = np.fromiter(
            (ase_atomic_numbers[str(symbol)] for symbol in self.atomname),
            dtype=np.int16,
            count=len(self.atomname),
        )

    def _parse_custom_cutoffs(self, cutoff_str):
        """Parse the custom cutoff string into a dictionary.

        Parameters
        ----------
        cutoff_str : str
            Custom cutoffs in the format "El1-El2:dist,El3-El4:dist"

        Returns
        -------
        dict
            Dictionary with frozenset pairs as keys and cutoffs as values
        """
        if not cutoff_str:
            return {}

        result = {}
        pairs = cutoff_str.split(",")
        for pair in pairs:
            pair = pair.strip()  # Clean input to allow spaces
            if not pair:  # Skip empty segments
                continue

            if ":" not in pair:
                raise ValueError(
                    f"Invalid custom cutoff format '{pair}'. Expected 'Element1-Element2:distance'. "
                    f"Example: 'Al-O:2.5,C-H:1.1'"
                )

            atom_pair_str, dist_str = pair.split(":", 1)
            atom_pair_str = atom_pair_str.strip()
            dist_str = dist_str.strip()

            if "-" not in atom_pair_str:
                raise ValueError(
                    f"Invalid custom cutoff format '{pair}'. Expected 'Element1-Element2:distance'. "
                    f"Example: 'Al-O:2.5,C-H:1.1'"
                )

            atom1, atom2 = atom_pair_str.split("-", 1)
            atom1 = atom1.strip()
            atom2 = atom2.strip()

            try:
                distance = float(dist_str)
            except ValueError:
                raise ValueError(
                    f"Invalid distance value '{dist_str}' in '{pair}'. Expected a number. "
                    f"Example: 'Al-O:2.5,C-H:1.1'"
                )

            result[frozenset({atom1, atom2})] = distance
        return result

    def _getbondfromase(self, step_atoms: Atoms, cell: np.ndarray):
        """Perceive bonds using ASE neighbor list with custom cutoffs.

        Parameters
        ----------
        step_atoms : ase.Atoms
            Atoms in a step.
        cell : np.ndarray
            Cell in the shape (3, 3).

        Returns
        -------
        list[list[int]]
            Connected atoms for each atom.
        list[list[int]]
            Bond levels (all set to 1 since ASE doesn't detect bond order).
        """
        step_atoms.set_cell(cell)
        step_atoms.set_pbc(self.pbc)
        if self.pbc:
            step_atoms.wrap()

        # Build per-atom natural cutoff radii (with global multiplier)
        radii = natural_cutoffs(step_atoms, mult=self.ase_cutoff_mult)
        symbols = step_atoms.get_chemical_symbols()

        # Build a per-symbol radius lookup (same symbol always has same radius)
        radii_by_symbol: dict = {}
        for sym, r in zip(symbols, radii):
            radii_by_symbol[sym] = r

        # Build complete per-pair cutoff dict for all unique element pairs.
        # ASE neighbor_list dict format: {('El1', 'El2'): distance}
        # Pairs NOT in the dict get cutoff=0, so we must cover ALL pairs.
        unique_symbols = sorted(set(symbols))
        cutoff_dict: dict = {}
        for s1 in unique_symbols:
            for s2 in unique_symbols:
                pair = frozenset({s1, s2})
                if pair in self._parsed_custom_cutoffs:
                    cutoff_dict[(s1, s2)] = self._parsed_custom_cutoffs[pair]
                else:
                    cutoff_dict[(s1, s2)] = radii_by_symbol[s1] + radii_by_symbol[s2]

        i_list, j_list = neighbor_list("ij", step_atoms, cutoff=cutoff_dict)

        atomnumber = len(step_atoms)
        bond: list = [[] for _ in range(atomnumber)]
        bondlevel: list = [[] for _ in range(atomnumber)]

        for i, j in zip(i_list, j_list):
            if i < j and j not in bond[i]:
                bond[i].append(j)
                bond[j].append(i)
                bondlevel[i].append(1)
                bondlevel[j].append(1)

        return bond, bondlevel

    def _new_openbabel_molecule(
        self,
        step_atoms: Atoms,
        cell: np.ndarray,
    ):
        """Create an Open Babel molecule while preserving input atom IDs."""
        mol = openbabel.OBMol()
        mol.BeginModify()
        for idx, (num, position) in enumerate(
            zip(step_atoms.get_atomic_numbers(), step_atoms.positions)
        ):
            atom = mol.NewAtom(idx)
            atom.SetAtomicNum(int(num))
            atom.SetVector(*position)
        if self.pbc:
            unit_cell = openbabel.OBUnitCell()
            unit_cell.SetData(
                openbabel.vector3(*cell[0]),
                openbabel.vector3(*cell[1]),
                openbabel.vector3(*cell[2]),
            )
            mol.CloneData(unit_cell)
            mol.SetPeriodicMol()
        return mol

    @staticmethod
    def _getbondlistsfromopenbabel(mol, atomnumber):
        """Convert one perceived Open Babel molecule to adjacency lists."""
        bond = [[] for _ in range(atomnumber)]
        bondlevel = [[] for _ in range(atomnumber)]
        for obbond in openbabel.OBMolBondIter(mol):
            atom1 = obbond.GetBeginAtom().GetId()
            atom2 = obbond.GetEndAtom().GetId()
            level = obbond.GetBondOrder()
            if level == 5:
                # aromatic, 5 in openbabel but 12 in rdkit
                level = 12
            bond[atom1].append(atom2)
            bond[atom2].append(atom1)
            bondlevel[atom1].append(level)
            bondlevel[atom2].append(level)
        return bond, bondlevel

    def _getbondfromopenbabel(
        self,
        step_atoms: Atoms,
        cell: np.ndarray,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Run the original Open Babel coordinate-bond perception path."""
        mol = self._new_openbabel_molecule(step_atoms, cell)
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        mol.EndModify()
        return self._getbondlistsfromopenbabel(mol, len(step_atoms))

    @staticmethod
    def _orthorhombic_cell_lengths(cell):
        """Return positive axis-aligned cell lengths when safely supported."""
        cell = np.asarray(cell, dtype=np.float64)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            return None
        lengths = np.diag(cell)
        if np.any(lengths <= 0.0) or not np.allclose(
            cell,
            np.diag(lengths),
            rtol=0.0,
            atol=1e-12,
        ):
            return None
        return lengths

    def _getbondfromperiodicneighborlist(
        self,
        step_atoms: Atoms,
        cell: np.ndarray,
    ) -> tuple[list[list[int]], list[list[int]]] | None:
        """Accelerate periodic Open Babel connectivity with bounded candidates.

        A periodic cKDTree supplies nearby atom pairs for axis-aligned cells,
        then Open Babel retains responsibility for insertion rules,
        valence/angle cleanup, and bond-order perception. Unsupported or
        numerically ambiguous geometries return ``None`` so the caller can use
        ``ConnectTheDots`` without changing its historical result.
        """
        atomnumber = len(step_atoms)
        if atomnumber < 2:
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_SMALL_FRAME
            return ([[] for _ in range(atomnumber)], [[] for _ in range(atomnumber)])

        positions = np.asarray(step_atoms.positions, dtype=np.float64)
        atomic_numbers = np.asarray(step_atoms.get_atomic_numbers())
        if (
            positions.shape != (atomnumber, 3)
            or not np.all(np.isfinite(positions))
            or atomic_numbers.shape != (atomnumber,)
        ):
            self._last_bond_detection_mode = (
                _BOND_DETECTION_OPENBABEL_INVALID_COORDINATES
            )
            return None
        radii = np.fromiter(
            (
                openbabel.GetCovalentRad(int(atomic_number))
                for atomic_number in atomic_numbers
            ),
            dtype=np.float64,
            count=atomnumber,
        )
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_INVALID_RADII
            return None

        cell_lengths = self._orthorhombic_cell_lengths(cell)
        maximum_pair_cutoff = (
            2.0 * float(np.max(radii)) + _OPENBABEL_COVALENT_RADIUS_TOLERANCE
        )
        if cell_lengths is None:
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_UNSUPPORTED_CELL
            return None
        if np.min(cell_lengths) <= 2.0 * (
            maximum_pair_cutoff + _OPENBABEL_DISTANCE_AMBIGUITY
        ):
            # Avoid duplicate lattice images and unbounded neighbor-list growth
            # in cells narrower than twice the largest possible bond distance.
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_NARROW_CELL
            return None

        z_order = np.argsort(positions[:, 2], kind="stable")
        # Open Babel compares only z and leaves equal-z ordering to the C++
        # standard library. Use atom ID as a stable tie-breaker so rounded
        # trajectory coordinates behave deterministically across platforms.

        wrapped_positions = np.mod(positions, cell_lengths)
        candidate_pairs = cKDTree(
            wrapped_positions,
            boxsize=cell_lengths,
        ).query_pairs(
            maximum_pair_cutoff + _OPENBABEL_DISTANCE_AMBIGUITY,
            output_type="ndarray",
        )
        atom_indices = candidate_pairs[:, 0]
        neighbor_indices = candidate_pairs[:, 1]
        distance_vectors = (
            wrapped_positions[neighbor_indices] - wrapped_positions[atom_indices]
        )
        distance_vectors -= np.rint(distance_vectors / cell_lengths) * cell_lengths
        distances = np.sqrt(np.einsum("ij,ij->i", distance_vectors, distance_vectors))

        pair_cutoffs = (
            radii[atom_indices]
            + radii[neighbor_indices]
            + _OPENBABEL_COVALENT_RADIUS_TOLERANCE
        )
        boundary_ambiguous = (
            np.abs(distances - pair_cutoffs) <= _OPENBABEL_DISTANCE_AMBIGUITY
        ) | (
            np.abs(distances - _OPENBABEL_MIN_BOND_DISTANCE)
            <= _OPENBABEL_DISTANCE_AMBIGUITY
        )
        if np.any(boundary_ambiguous):
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY
            return None

        accepted = (distances <= pair_cutoffs) & (
            distances >= _OPENBABEL_MIN_BOND_DISTANCE
        )
        atom_indices = atom_indices[accepted]
        neighbor_indices = neighbor_indices[accepted]

        z_rank = np.empty(atomnumber, dtype=np.intp)
        z_rank[z_order] = np.arange(atomnumber, dtype=np.intp)
        reverse_pair = z_rank[atom_indices] > z_rank[neighbor_indices]
        first_atoms = np.where(reverse_pair, neighbor_indices, atom_indices)
        second_atoms = np.where(reverse_pair, atom_indices, neighbor_indices)
        insertion_order = np.lexsort((z_rank[second_atoms], z_rank[first_atoms]))

        mol = self._new_openbabel_molecule(step_atoms, cell)
        for atom_index, neighbor_index in zip(
            first_atoms[insertion_order],
            second_atoms[insertion_order],
        ):
            atom = mol.GetAtom(int(atom_index) + 1)
            neighbor = mol.GetAtom(int(neighbor_index) + 1)
            if atom.GetAtomicNum() == 15 and atom.GetExplicitValence() == 5:
                if neighbor.GetAtomicNum() not in (9, 17):
                    continue
            if neighbor.GetAtomicNum() == 15 and neighbor.GetExplicitValence() == 5:
                if atom.GetAtomicNum() not in (9, 17):
                    continue
            mol.AddBond(int(atom_index) + 1, int(neighbor_index) + 1, 1)

        for atom in openbabel.OBMolAtomIter(mol):
            while (
                atom.GetExplicitValence() > openbabel.GetMaxBonds(atom.GetAtomicNum())
                or atom.SmallestBondAngle() < _OPENBABEL_MIN_BOND_ANGLE
            ):
                bonds = list(openbabel.OBAtomBondIter(atom))
                if not bonds:
                    break
                if atom.GetAtomicNum() == 1:
                    hydrogen_bond = next(
                        (
                            bond
                            for bond in bonds
                            if bond.GetNbrAtom(atom).GetAtomicNum() == 1
                        ),
                        None,
                    )
                    if hydrogen_bond is not None:
                        mol.DeleteBond(hydrogen_bond)
                        continue
                longest_bond = bonds[0]
                longest_length = longest_bond.GetLength()
                for bond in bonds[1:]:
                    length = bond.GetLength()
                    if length > longest_length:
                        longest_bond = bond
                        longest_length = length
                mol.DeleteBond(longest_bond)

        mol.PerceiveBondOrders()
        mol.EndModify()
        self._last_bond_detection_mode = _BOND_DETECTION_CKDTREE
        return self._getbondlistsfromopenbabel(mol, atomnumber)

    def _getbondfromcrd(
        self, step_atoms: Atoms, cell: np.ndarray
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Perceive bonds from atomic coordinates.

        Parameters
        ----------
        step_atoms : ase.Atoms
            Atoms in a step.
        cell : np.ndarray
            Cell in the shape (3, 3).

        Returns
        -------
        list[list[int]]
            Connected atoms for each atom.
        list[list[int]]
            Bond orders for each atom. 12 is an aromatic bond.
        """
        # Check if ASE mode is enabled
        if self.use_ase:
            self._last_bond_detection_mode = _BOND_DETECTION_ASE
            return self._getbondfromase(step_atoms, cell)

        atomnumber = len(step_atoms)
        if not self.pbc:
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_NONPERIODIC
        elif atomnumber < _OPENBABEL_PERIODIC_NEIGHBOR_MIN_ATOMS:
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_SMALL_FRAME
        elif self._orthorhombic_cell_lengths(cell) is None:
            self._last_bond_detection_mode = _BOND_DETECTION_OPENBABEL_UNSUPPORTED_CELL
        else:
            accelerated = self._getbondfromperiodicneighborlist(step_atoms, cell)
            if accelerated is not None:
                self._last_bond_detection_mode = _BOND_DETECTION_CKDTREE
                return accelerated
        return self._getbondfromopenbabel(step_atoms, cell)


@_Detect.register_subclass("dump")
@_Detect.register_subclass("lammpsdumpfile")
class _DetectLAMMPSdump(_DetectCrd):
    class LineType(Enum):
        """Line type in the LAMMPS dump files."""

        TIMESTEP = auto()
        ATOMS = auto()
        NUMBER = auto()
        BOX = auto()
        OTHER = auto()

        @classmethod
        def linecontent(cls, line):
            """Return line content."""
            if line.startswith("ITEM: TIMESTEP"):
                return cls.TIMESTEP
            if line.startswith("ITEM: ATOMS"):
                return cls.ATOMS
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                return cls.NUMBER
            if line.startswith("ITEM: BOX"):
                return cls.BOX
            return cls.OTHER

    def _readNfunc(self, f):
        iscompleted = False
        N = None
        linecontent = None
        atomtype = None
        index = -1
        stepaindex = None
        for index, line in enumerate(f):
            if line.startswith("ITEM:"):
                linecontent = self.LineType.linecontent(line)
                if linecontent == self.LineType.ATOMS:
                    keys = line.split()
                    self.id_idx = keys.index("id") - 2
                    self.tidx = keys.index("type") - 2
                    self.xidx = keys.index("x") - 2
                    self.yidx = keys.index("y") - 2
                    self.zidx = keys.index("z") - 2
            else:
                assert linecontent is not None
                if linecontent == self.LineType.NUMBER:
                    if iscompleted:
                        assert stepaindex is not None
                        steplinenum = index - stepaindex
                        break
                    else:
                        iscompleted = True
                        stepaindex = index
                    N = int(line.split()[0])
                    atomtype = np.zeros(N, dtype=int)
                elif linecontent == self.LineType.ATOMS:
                    s = line.split()
                    assert atomtype is not None
                    atomtype[int(s[self.id_idx]) - 1] = int(s[self.tidx]) - 1
        else:
            steplinenum = index + 1
        assert N is not None and atomtype is not None
        self.N = N
        self.atomtype = atomtype
        return steplinenum

    def _readstepfunc(self, item) -> tuple[list[bytes], tuple[int, int]]:
        step, lines = item
        step_atomic_numbers = np.empty(self.N, dtype=np.int16)
        step_positions = np.empty((self.N, 3), dtype=np.float64)
        seen_atom_ids = np.zeros(self.N, dtype=np.bool_)
        ss = []
        linecontent = None
        timestep = None
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    linecontent = self.LineType.linecontent(line)
                else:
                    if linecontent is None:
                        raise ValueError("LAMMPS dump file format error")
                    elif linecontent == self.LineType.ATOMS:
                        s = line.split()
                        atom_index = int(s[self.id_idx]) - 1
                        atom_type = int(s[self.tidx]) - 1
                        if atom_index < 0 or atom_index >= self.N:
                            raise ValueError("LAMMPS dump atom ID is out of range")
                        if seen_atom_ids[atom_index]:
                            raise ValueError("LAMMPS dump contains a duplicate atom ID")
                        if atom_type < 0 or atom_type >= len(
                            self._atomic_numbers_by_type
                        ):
                            raise ValueError("LAMMPS dump atom type is out of range")
                        step_atomic_numbers[atom_index] = self._atomic_numbers_by_type[
                            atom_type
                        ]
                        step_positions[atom_index] = (
                            float(s[self.xidx]),
                            float(s[self.yidx]),
                            float(s[self.zidx]),
                        )
                        seen_atom_ids[atom_index] = True
                    elif linecontent == self.LineType.TIMESTEP:
                        timestep = step, int(line.split()[0])
                    elif linecontent == self.LineType.BOX:
                        s = line.split()
                        ss.append(list(map(float, s)))
        assert timestep is not None
        ss = np.array(ss)
        if ss.shape[1] > 2:
            xy = ss[0][2]
            xz = ss[1][2]
            yz = ss[2][2]
        else:
            xy, xz, yz = 0.0, 0.0, 0.0
        xlo = ss[0][0] - min(0.0, xy, xz, xy + xz)
        xhi = ss[0][1] - max(0.0, xy, xz, xy + xz)
        ylo = ss[1][0] - min(0.0, yz)
        yhi = ss[1][1] - max(0.0, yz)
        zlo = ss[2][0]
        zhi = ss[2][1]
        boxsize = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [xy, yhi - ylo, 0.0],
                [xz, yz, zhi - zlo],
            ]
        )
        if not np.all(seen_atom_ids):
            raise ValueError("LAMMPS dump frame is missing one or more atom IDs")
        step_atoms = Atoms(numbers=step_atomic_numbers, positions=step_positions)
        bond, level = self._getbondfromcrd(step_atoms, boxsize)
        molecules = self._connectmolecule(bond, level)
        return molecules, timestep


@_Detect.register_subclass("xyz")
class _Detectxyz(_DetectCrd):
    """xyz file. Two frames are connected. Cell information must be inputed by user."""

    def _readNfunc(self, f):
        atomname_dict = dict(zip(self.atomname.tolist(), range(self.atomname.size)))
        N = None
        atomtype = None
        for index, line in enumerate(f):
            s = line.split()
            if index == 0:
                N = int(line.strip())
                atomtype = np.zeros(N, dtype=int)
            elif N is None:
                assert N is not None  # pragma: no cover
            elif index > N + 1:
                break
            elif index > 1:
                assert atomtype is not None
                atomtype[index - 2] = atomname_dict[s[0]]
        assert N is not None and atomtype is not None
        self.N = N
        self.atomtype = atomtype
        steplinenum = N + 2
        return steplinenum

    def _readstepfunc(self, item) -> tuple[list[bytes], tuple[int, int]]:
        step, lines = item
        timestep = step, step
        step_atoms = []
        boxsize = self.cell
        if self.pbc and boxsize is None:
            raise RuntimeError("No cell information is given.")
        for index, line in enumerate(lines):
            s = line.split()
            if index > 1:
                step_atoms.append(
                    (index - 1, Atom(s[0], tuple(float(x) for x in s[1:4])))
                )
        _, step_atoms = zip(*sorted(step_atoms, key=operator.itemgetter(0)))
        step_atoms = Atoms(step_atoms)
        bond, level = self._getbondfromcrd(step_atoms, boxsize)
        molecules = self._connectmolecule(bond, level)
        return molecules, timestep


@_Detect.register_subclass("extxyz")
class _Detectextxyz(_DetectCrd):
    """extxyz file. xyz with extended metadata support like cell, force, etc."""

    def _readNfunc(self, f):
        atomname_dict = dict(zip(self.atomname.tolist(), range(self.atomname.size)))
        N = None
        atomtype = None
        for index, line in enumerate(f):
            if index == 0:
                N = int(line.strip())
                atomtype = np.zeros(N, dtype=int)
            elif index == 1:
                # Extract lattice (cell) from comment line
                if "Lattice=" in line:
                    lattice_str = line.split("Lattice=")[1].split('"')[1]
                    lattice_floats = list(map(float, lattice_str.split()))
                    self.cell = np.array(lattice_floats).reshape((3, 3))
                else:
                    raise RuntimeError("Missing Lattice= information in extxyz.")
            elif (N is not None) and (index > N + 1):
                break
            elif index > 1:
                s = line.split()
                assert atomtype is not None
                atomtype[index - 2] = atomname_dict[s[0]]
        assert N is not None and atomtype is not None
        self.N = N
        self.atomtype = atomtype
        steplinenum = N + 2
        return steplinenum

    def _readstepfunc(self, item) -> tuple[list[bytes], tuple[int, int]]:
        step, lines = item
        step_atoms = []
        timestep = step, step  # Use step as timestep fallback
        boxsize = self.cell
        if self.pbc and boxsize is None:
            raise RuntimeError("No cell information is given in extxyz.")
        for index, line in enumerate(lines):
            if index > 1:
                s = line.split()
                step_atoms.append(
                    (index - 1, Atom(s[0], tuple(float(x) for x in s[1:4])))
                )
        _, step_atoms = zip(*sorted(step_atoms, key=operator.itemgetter(0)))
        step_atoms = Atoms(step_atoms)
        bond, level = self._getbondfromcrd(step_atoms, boxsize)
        molecules = self._connectmolecule(bond, level)
        return molecules, timestep
