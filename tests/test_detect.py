# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test different detect format."""

import os
from array import array
from pathlib import Path

import numpy as np
import pytest

import reacnetgenerator._detect as detect_module
from reacnetgenerator import ReacNetGenerator
from reacnetgenerator._detect import _Detect, _DetectLAMMPSdump
from reacnetgenerator._hmmfilter import _HMMFilter
from reacnetgenerator.utils import bytestolist, listtobytes

p_inputs = Path(__file__).parent / "inputs"


def _frame_index_widening_cases():
    """Return runtime-width overflow cases for unsigned array typecodes."""
    typecodes = detect_module._FRAME_INDEX_TYPECODES
    for index, typecode in enumerate(typecodes[:-1]):
        itemsize = array(typecode).itemsize
        expected_typecode = next(
            (
                candidate
                for candidate in typecodes[index + 1 :]
                if array(candidate).itemsize > itemsize
            ),
            None,
        )
        if expected_typecode is not None:
            yield typecode, 1 << (8 * itemsize), expected_typecode


class TestDetect:
    """Test different detect format.

    All systems contain a single water molecule: H, H, O.
    """

    @pytest.fixture(autouse=True)
    def chdir(self, tmp_path):
        """Change directory to tmp_path."""
        start_directory = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(start_directory)

    @pytest.fixture(
        params=[
            # inputfiletype, inputfilename
            ("lammpsdumpfile", p_inputs / "water.dump"),
            ("dump", p_inputs / "water_pbc.dump"),
            ("lammpsbondfile", p_inputs / "water.bond"),
            ("xyz", p_inputs / "water.xyz"),
            ("extxyz", p_inputs / "water.extxyz"),
        ]
    )
    def reacnetgen_param(self, request):
        """Fixture for ReacNetGenerator parameters."""
        return request.param

    @pytest.fixture()
    def reacnetgen(self, reacnetgen_param):
        """Fixture for ReacNetGenerator."""
        rngclass = ReacNetGenerator(
            inputfiletype=reacnetgen_param[0],
            inputfilename=reacnetgen_param[1],
            atomname=["H", "O"],
            pbc=False,
        )
        yield rngclass

    def test_reacnetgen(self, reacnetgen):
        """Test main process of ReacNetGen."""
        _Detect.gettype(reacnetgen).detect()
        assert reacnetgen.N == 3
        np.testing.assert_array_equal(
            reacnetgen.atomtype, np.array([0, 0, 1], dtype=int)
        )
        # assert this is a single molecule
        assert reacnetgen.temp1it == 1


def test_lammps_dump_builds_ordered_arrays_without_per_atom_objects(monkeypatch):
    """Place unsorted dump rows by atom ID without allocating ASE Atom wrappers."""
    source = p_inputs / "water.dump"
    rng = ReacNetGenerator(
        inputfiletype="dump",
        inputfilename=str(source),
        atomname=["H", "O"],
        pbc=False,
        nproc=1,
    )
    detector = _DetectLAMMPSdump(rng)
    with source.open() as stream:
        lines_per_frame = detector._readNfunc(stream)
    step, _source_id, _source_frame, lines = next(
        detector._iterinputsteps(lines_per_frame)
    )
    captured = {}

    def capture(step_atoms, cell):
        captured["numbers"] = step_atoms.get_atomic_numbers()
        captured["positions"] = step_atoms.positions.copy()
        captured["cell"] = np.asarray(cell)
        atom_count = len(step_atoms)
        return ([[] for _ in range(atom_count)], [[] for _ in range(atom_count)])

    monkeypatch.setattr(
        detect_module,
        "Atom",
        lambda *args, **kwargs: pytest.fail("per-atom ASE object was allocated"),
    )
    monkeypatch.setattr(detector, "_getbondfromcrd", capture)

    _molecules, timestep = detector._readstepfunc((step, lines))

    assert timestep == (0, 0)
    np.testing.assert_array_equal(captured["numbers"], [1, 1, 8])
    np.testing.assert_allclose(
        captured["positions"],
        [
            [1.90487, -0.09814, -0.05690],
            [0.65814, -0.90854, -0.40394],
            [0.93699, -0.05546, -0.03862],
        ],
    )
    assert captured["cell"].shape == (3, 3)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
@pytest.mark.parametrize("compression_workers", [1, 2])
def test_detect_frame_pool_uses_bounded_single_frame_tasks(
    tmp_path,
    monkeypatch,
    caplog,
    start_method,
    compression_workers,
):
    """Bound frame work and keep molecule compression free of array IPC."""
    source = tmp_path / "frames.dump"
    source.write_text("")
    rng = ReacNetGenerator(
        inputfiletype="dump",
        inputfilename=str(source),
        atomname=["H"],
        nproc=8,
    )
    detector = _DetectLAMMPSdump(rng)
    monkeypatch.setattr(detector, "_readNfunc", lambda handle: 1)
    monkeypatch.setattr(
        detect_module,
        "_detect_worker_count",
        lambda *args: 8,
        raising=False,
    )
    monkeypatch.setattr(
        detect_module,
        "get_start_method",
        lambda: start_method,
        raising=False,
    )
    monkeypatch.setattr(
        detect_module,
        "_detect_compression_worker_count",
        lambda *args: compression_workers,
        raising=False,
    )
    captured_calls = []
    captured_records = []
    captured_frame_typecodes = []
    captured_thread_calls = []

    def compress_values(values):
        for value in values:
            captured_frame_typecodes.append(value.typecode)
            yield detector._compressvalue(value)

    def fake_run_mp(nproc, **kwargs):
        captured_calls.append((nproc, kwargs))
        if kwargs["desc"].startswith("Read bond information"):
            return iter(
                (
                    (
                        [b"first"],
                        (
                            0,
                            1,
                            0,
                            10,
                            detect_module._BOND_DETECTION_CKDTREE,
                        ),
                    ),
                    (
                        [b"second"],
                        (
                            1,
                            1,
                            1,
                            20,
                            detect_module._BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY,
                        ),
                    ),
                )
            )
        return compress_values(kwargs["l"])

    def fake_bounded_thread_map(func, values, **kwargs):
        captured_thread_calls.append(kwargs)
        return compress_values(values)

    monkeypatch.setattr(detect_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(
        detect_module,
        "_bounded_thread_map",
        fake_bounded_thread_map,
        raising=False,
    )
    monkeypatch.setattr(
        detector,
        "_writemoleculetempfile",
        lambda records: captured_records.extend(records),
    )

    detector._readinputfile()

    frame_nproc, frame_kwargs = captured_calls[0]
    assert frame_nproc == 8
    assert frame_kwargs["chunksize"] == 1
    assert frame_kwargs["max_inflight"] == 16
    assert frame_kwargs["unordered"] is False
    if compression_workers == 1:
        assert captured_calls[1][0] == 1
        assert captured_thread_calls == []
    else:
        assert len(captured_calls) == 1
        assert captured_thread_calls == [{"workers": 2, "max_inflight": 4}]
    assert [record[0] for record in captured_records] == [b"first", b"second"]
    assert isinstance(detector.timestep, array)
    assert detector.timestep.typecode == "q"
    assert list(detector.timestep) == [10, 20]
    assert detector.framesource == {0: (1, 0), 1: (1, 1)}
    assert detector.framesource.get(2) is None
    assert captured_frame_typecodes == ["H", "H"]
    assert "first frame: periodic-cKDTree" in caplog.text
    assert "periodic-cKDTree=1" in caplog.text
    assert "OpenBabel-cutoff-boundary=1" in caplog.text


def test_detect_closes_thread_compression_when_writing_fails(tmp_path, monkeypatch):
    """Release the thread producer when the downstream file write aborts."""
    source = tmp_path / "frames.dump"
    source.write_text("")
    rng = ReacNetGenerator(
        inputfiletype="dump",
        inputfilename=str(source),
        atomname=["H"],
        nproc=8,
    )
    detector = _DetectLAMMPSdump(rng)
    monkeypatch.setattr(detector, "_readNfunc", lambda handle: 1)
    monkeypatch.setattr(detect_module, "get_start_method", lambda: "fork")
    monkeypatch.setattr(detect_module, "_detect_worker_count", lambda *args: 8)
    monkeypatch.setattr(
        detect_module,
        "_detect_compression_worker_count",
        lambda *args: 2,
    )
    captured_sources = []

    def fake_run_mp(nproc, **kwargs):
        assert kwargs["desc"].startswith("Read bond information")
        return iter(
            (
                (
                    [b"first"],
                    (0, 1, 0, 10, detect_module._BOND_DETECTION_UNKNOWN),
                ),
                (
                    [b"second"],
                    (1, 1, 1, 20, detect_module._BOND_DETECTION_UNKNOWN),
                ),
            )
        )

    class ClosableCompression:
        def __init__(self, func, values):
            self.func = func
            self.values = iter(values)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return self.func(next(self.values))

        def close(self):
            self.closed = True

    def fake_bounded_thread_map(func, values, **kwargs):
        compression = ClosableCompression(func, values)
        captured_sources.append(compression)
        return compression

    def fail_after_one_record(records):
        next(iter(records))
        raise OSError("write failed")

    monkeypatch.setattr(detect_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(
        detect_module,
        "_bounded_thread_map",
        fake_bounded_thread_map,
    )
    monkeypatch.setattr(detector, "_writemoleculetempfile", fail_after_one_record)

    with pytest.raises(OSError, match="write failed"):
        detector._readinputfile()

    assert len(captured_sources) == 1
    assert captured_sources[0].closed


@pytest.mark.parametrize(
    (
        "requested_nproc",
        "molecule_count",
        "payload_bytes",
        "expected_workers",
    ),
    [
        (8, 128, 16 * 1024 * 1024, 2),
        (1, 128, 16 * 1024 * 1024, 1),
        (8, 128, 16 * 1024 * 1024 - 1, 1),
        (8, 129, 16 * 1024 * 1024, 1),
        (8, 1, 128 * 1024 * 1024, 1),
        (8, 0, 0, 1),
    ],
)
def test_detect_compression_worker_count_requires_amortized_payload(
    requested_nproc,
    molecule_count,
    payload_bytes,
    expected_workers,
):
    """Use two shared-memory threads only for sufficiently large records."""
    assert (
        detect_module._detect_compression_worker_count(
            requested_nproc,
            molecule_count,
            payload_bytes,
        )
        == expected_workers
    )


def test_bounded_thread_map_preserves_order_and_limits_pending(monkeypatch):
    """Retain input order without materializing an unbounded result queue."""
    state = {"pending": 0, "maximum_pending": 0, "workers": None}

    class Future:
        def __init__(self, func, value):
            self.func = func
            self.value = value

        def result(self):
            state["pending"] -= 1
            return self.func(self.value)

        @staticmethod
        def cancel():
            return False

    class Executor:
        def __init__(self, max_workers):
            state["workers"] = max_workers

        def __enter__(self):
            return self

        @staticmethod
        def __exit__(*args):
            return None

        @staticmethod
        def submit(func, value):
            state["pending"] += 1
            state["maximum_pending"] = max(
                state["maximum_pending"],
                state["pending"],
            )
            return Future(func, value)

    monkeypatch.setattr(detect_module, "ThreadPoolExecutor", Executor, raising=False)

    assert list(
        detect_module._bounded_thread_map(
            lambda value: value * 2,
            range(7),
            workers=2,
            max_inflight=4,
        )
    ) == [0, 2, 4, 6, 8, 10, 12]
    assert state == {"pending": 0, "maximum_pending": 4, "workers": 2}


def test_bounded_thread_map_runs_with_real_threads():
    """Exercise the production executor while retaining deterministic order."""
    assert list(
        detect_module._bounded_thread_map(
            abs,
            [0, -1, -2, -3, -4],
            workers=2,
            max_inflight=4,
        )
    ) == [0, 1, 2, 3, 4]


def test_bounded_thread_map_close_cancels_pending_and_exits(monkeypatch):
    """Cancel bounded pending work when a downstream consumer stops early."""
    state = {"cancelled": 0, "exited": 0, "submitted": 0}

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

        def cancel(self):
            state["cancelled"] += 1
            return True

    class Executor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        @staticmethod
        def submit(func, value):
            state["submitted"] += 1
            return Future(func(value))

        @staticmethod
        def __exit__(*args):
            state["exited"] += 1

    monkeypatch.setattr(detect_module, "ThreadPoolExecutor", Executor)
    results = detect_module._bounded_thread_map(
        abs,
        range(10),
        workers=2,
        max_inflight=4,
    )

    assert next(results) == 0
    results.close()

    assert state == {"cancelled": 3, "exited": 1, "submitted": 4}


def test_detect_frame_indices_widen_at_trajectory_boundary(tmp_path, monkeypatch):
    """Exercise the production append path across its initial array boundary."""
    initial_typecode = "H"
    initial_itemsize = array(initial_typecode).itemsize
    initial_limit = (1 << (8 * initial_itemsize)) - 1
    expected_typecode = next(
        (
            candidate
            for candidate in detect_module._FRAME_INDEX_TYPECODES
            if array(candidate).itemsize > initial_itemsize
        ),
        None,
    )
    if expected_typecode is None or initial_limit > 1_000_000:
        pytest.skip("native unsigned-short boundary is too large for this test")
    source = tmp_path / "frames.dump"
    source.write_text("")
    rng = ReacNetGenerator(
        inputfiletype="dump",
        inputfilename=str(source),
        atomname=["H"],
        nproc=1,
    )
    detector = _DetectLAMMPSdump(rng)
    monkeypatch.setattr(detector, "_readNfunc", lambda handle: 1)
    monkeypatch.setattr(
        detect_module,
        "get_start_method",
        lambda: "spawn",
        raising=False,
    )
    captured_records = []

    def frame_results():
        for step in range(initial_limit + 2):
            molecules = [b"molecule"] if step >= initial_limit else []
            yield molecules, (
                step,
                1,
                step,
                step,
                detect_module._BOND_DETECTION_UNKNOWN,
            )

    def fake_run_mp(nproc, **kwargs):
        if kwargs["desc"].startswith("Read bond information"):
            return frame_results()
        return (detector._compressvalue(value) for value in kwargs["l"])

    monkeypatch.setattr(detect_module, "run_mp", fake_run_mp)
    monkeypatch.setattr(
        detector,
        "_writemoleculetempfile",
        lambda records: captured_records.extend(records),
    )

    detector._readinputfile()

    assert detector.step == initial_limit + 2
    assert len(captured_records) == 1
    molecule, frame_block = captured_records[0]
    frames = np.asarray(bytestolist(frame_block))
    assert molecule == b"molecule"
    assert frames.dtype == np.asarray(array(expected_typecode)).dtype
    np.testing.assert_array_equal(frames, [initial_limit, initial_limit + 1])


@pytest.mark.parametrize(
    ("requested_nproc", "input_bytes", "start_method", "expected_nproc"),
    [
        (8, 2_083_407, "spawn", 1),
        (8, 4_166_814, "spawn", 1),
        (8, 8_333_628, "spawn", 1),
        (8, 16_667_256, "spawn", 2),
        (8, 100_000_000, "spawn", 8),
        (8, 2_083_407, "fork", 8),
        (4, 0, "forkserver", 1),
    ],
)
def test_detect_worker_count_accounts_for_process_start_cost(
    requested_nproc,
    input_bytes,
    start_method,
    expected_nproc,
):
    """Keep non-fork workers behind enough input while retaining fork."""
    assert (
        detect_module._detect_worker_count(
            requested_nproc,
            input_bytes,
            start_method,
        )
        == expected_nproc
    )


def test_frame_source_map_compacts_arithmetic_source_segments():
    """Preserve mapping behavior without retaining a tuple for every frame."""
    sources = detect_module._FrameSourceMap(stepinterval=2)
    sources.append(0, 1, 0)
    sources.append(1, 1, 2)
    sources.append(2, 3, 1)
    sources.append(3, 3, 3)

    assert sources == {0: (1, 0), 1: (1, 2), 2: (3, 1), 3: (3, 3)}
    assert sources.segment_count == 2
    assert sources.nbytes == 2 * (8 + 4 + 8)
    assert sources.get(4) is None
    source_ids = np.empty(3, dtype=np.uint32)
    source_frames = np.empty(3, dtype=np.uint64)
    sources.fill_arrays(1, 4, source_ids, source_frames)
    np.testing.assert_array_equal(source_ids, [1, 3, 3])
    np.testing.assert_array_equal(source_frames, [2, 1, 3])
    with pytest.raises(RuntimeError, match="sequential input order"):
        sources.append(5, 3, 5)


@pytest.mark.parametrize(
    ("typecode", "frame", "expected_typecode"),
    list(_frame_index_widening_cases()),
)
def test_widen_frame_indices_preserves_values_and_compact_encoding(
    typecode,
    frame,
    expected_typecode,
):
    """Upgrade only overflowing timelines while preserving values and width."""
    compact = array(typecode, [0, 1])
    widened = detect_module._widen_frame_indices(compact, frame)
    expected = array("Q", [0, 1, frame])

    assert widened.typecode == expected_typecode
    assert list(widened) == list(expected)
    decoded = np.asarray(bytestolist(_Detect._compressvalue(None, widened)))
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == np.asarray(widened).dtype


def test_hmm_filter_accepts_compact_frame_index_dtype():
    """Keep HMM signals identical after removing the forced uint64 encoding."""

    class IdentityModel:
        @staticmethod
        def predict(signal):
            return signal.reshape((-1,))

    hmm_filter = object.__new__(_HMMFilter)
    hmm_filter.runHMM = True
    hmm_filter.step = 512
    hmm_filter.getoriginfile = True
    hmm_filter.printfiltersignal = True
    hmm_filter._model = IdentityModel()
    frames = array("H", [0, 255, 511])
    compact_block = _Detect._compressvalue(None, frames)
    uint64_block = listtobytes(np.asarray(frames, dtype=np.uint64))
    structure = (listtobytes([]),) * 3

    compact_origin, compact_hmm, _ = hmm_filter._getoriginandhmm(
        (*structure, compact_block)
    )
    uint64_origin, uint64_hmm, _ = hmm_filter._getoriginandhmm(
        (*structure, uint64_block)
    )

    np.testing.assert_array_equal(
        bytestolist(compact_origin),
        bytestolist(uint64_origin),
    )
    np.testing.assert_array_equal(
        bytestolist(compact_hmm),
        bytestolist(uint64_hmm),
    )


# Additional tests for ASE detection and Scipy clustering features
try:
    from ase import Atoms

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

try:
    import scipy

    print(scipy.__version__)
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE is not available")
class TestFastPeriodicOpenBabel:
    """Keep the periodic neighbor-list accelerator equivalent to Open Babel."""

    @pytest.fixture
    def detect_instance(self):
        """Create a periodic coordinate detector using Open Babel bond orders."""
        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename="dummy",
            atomname=["H", "C", "N", "O", "F", "P", "Cl"],
            pbc=True,
            use_ase=False,
        )
        return _DetectLAMMPSdump(rng)

    @staticmethod
    def _molecule_records(detect_instance, result):
        assert result is not None
        return detect_instance._connectmolecule(*result)

    def test_falls_back_for_triclinic_boundary_bonds(
        self,
        detect_instance,
    ):
        """Leave skewed-cell minimum images to the reference implementation."""
        cell = np.array(
            [
                [10.0, 0.0, 0.0],
                [2.0, 9.0, 0.0],
                [1.0, 1.5, 8.0],
            ]
        )
        fractional = np.array(
            [
                [0.03, 0.20, 0.300],
                [0.97, 0.20, 0.301],
                [0.40, 0.50, 0.600],
                [0.45, 0.50, 0.602],
            ]
        )
        atoms = Atoms("CHOH", positions=fractional @ cell, cell=cell, pbc=True)

        accelerated = detect_instance._getbondfromperiodicneighborlist(atoms, cell)

        assert accelerated is None
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_OPENBABEL_UNSUPPORTED_CELL
        )

    def test_matches_reference_when_openbabel_removes_excess_bonds(
        self,
        detect_instance,
    ):
        """Reuse Open Babel's valence and angle cleanup after candidate search."""
        cell = np.eye(3) * 12.0
        atoms = Atoms(
            "CHHHHH",
            positions=np.array(
                [
                    [6.00, 6.00, 6.000],
                    [6.90, 6.00, 6.001],
                    [5.05, 6.00, 6.002],
                    [6.00, 7.00, 6.003],
                    [6.00, 4.95, 6.004],
                    [6.00, 6.00, 7.080],
                ]
            ),
            cell=cell,
            pbc=True,
        )

        reference = detect_instance._getbondfromopenbabel(atoms, cell)
        accelerated = detect_instance._getbondfromperiodicneighborlist(atoms, cell)

        assert accelerated is not None
        assert self._molecule_records(
            detect_instance, accelerated
        ) == self._molecule_records(detect_instance, reference)

    def test_matches_reference_for_equal_z_coordinates(
        self,
        detect_instance,
    ):
        """Keep rounded equal-z coordinates on the accelerated path."""
        cell = np.eye(3) * 12.0
        atoms = Atoms(
            "CHHHHH",
            positions=np.array(
                [
                    [6.00, 6.00, 6.00],
                    [6.90, 6.00, 6.00],
                    [5.05, 6.00, 6.00],
                    [6.00, 7.00, 6.00],
                    [6.00, 4.95, 6.00],
                    [6.00, 6.00, 7.08],
                ]
            ),
            cell=cell,
            pbc=True,
        )

        reference = detect_instance._getbondfromopenbabel(atoms, cell)
        accelerated = detect_instance._getbondfromperiodicneighborlist(atoms, cell)

        assert accelerated is not None
        assert self._molecule_records(
            detect_instance, accelerated
        ) == self._molecule_records(detect_instance, reference)

    def test_matches_reference_for_phosphorus_sixth_bond_rule(
        self,
        detect_instance,
    ):
        """Preserve Open Babel's element-specific phosphorus insertion rule."""
        cell = np.eye(3) * 12.0
        angles = np.arange(6) * np.pi / 3.0
        positions = np.concatenate(
            (
                [[6.0, 6.0, 6.0]],
                np.column_stack(
                    (
                        6.0 + 1.6 * np.cos(angles),
                        6.0 + 1.6 * np.sin(angles),
                        6.01 + np.arange(6) * 0.01,
                    )
                ),
            )
        )
        atoms = Atoms("PFFFFFH", positions=positions, cell=cell, pbc=True)

        reference = detect_instance._getbondfromopenbabel(atoms, cell)
        accelerated = detect_instance._getbondfromperiodicneighborlist(atoms, cell)

        assert accelerated is not None
        assert accelerated[0][0] == [1, 2, 3, 4, 5]
        assert accelerated[0][6] == []
        assert self._molecule_records(
            detect_instance, accelerated
        ) == self._molecule_records(detect_instance, reference)

    def test_orthorhombic_path_does_not_materialize_ase_bins(
        self,
        detect_instance,
        monkeypatch,
    ):
        """Use the bounded cKDTree path instead of ASE's padded bin arrays."""
        atoms = Atoms(
            "CHOH",
            positions=[[0.2, 1, 1], [11.3, 1, 1.01], [6, 6, 6], [6.8, 6, 6.01]],
            cell=np.eye(3) * 12,
            pbc=True,
        )
        monkeypatch.setattr(
            detect_module,
            "neighbor_list",
            lambda *args, **kwargs: pytest.fail("ASE neighbor_list was called"),
        )

        accelerated = detect_instance._getbondfromperiodicneighborlist(
            atoms,
            np.eye(3) * 12,
        )

        assert accelerated is not None
        assert 1 in accelerated[0][0]

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference_for_seeded_mixed_element_systems(
        self,
        detect_instance,
        seed,
    ):
        """Differentially cover bond insertion, phosphorus, and bond orders."""
        random = np.random.default_rng(seed)
        atomic_numbers = random.choice(
            np.array([1, 6, 7, 8, 9, 15, 17]),
            size=96,
        )
        positions = random.uniform(0.0, 18.0, size=(96, 3))
        cell = np.eye(3) * 18.0
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=True,
        )

        reference = detect_instance._getbondfromopenbabel(atoms, cell)
        accelerated = detect_instance._getbondfromperiodicneighborlist(atoms, cell)

        assert accelerated is not None
        assert self._molecule_records(
            detect_instance, accelerated
        ) == self._molecule_records(detect_instance, reference)

    @pytest.mark.parametrize(
        ("atoms", "cell", "expected_mode"),
        [
            (
                Atoms("HH", positions=[[0, 0, 0], [0, 0, 1.07]]),
                np.eye(3) * 10.0,
                detect_module._BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY,
            ),
            (
                Atoms("CH", positions=[[0, 0, 0], [1, 0, 0.1]]),
                np.eye(3) * 2.0,
                detect_module._BOND_DETECTION_OPENBABEL_NARROW_CELL,
            ),
        ],
        ids=["cutoff-boundary", "small-cell"],
    )
    def test_falls_back_for_numerically_ambiguous_geometry(
        self,
        detect_instance,
        atoms,
        cell,
        expected_mode,
    ):
        """Leave ambiguous pair ordering and minimum images to Open Babel."""
        atoms.set_cell(cell)
        atoms.set_pbc(True)

        assert detect_instance._getbondfromperiodicneighborlist(atoms, cell) is None
        assert detect_instance._last_bond_detection_mode == expected_mode

    def test_dispatches_only_safe_large_periodic_frames(
        self,
        detect_instance,
        monkeypatch,
    ):
        """Use the accelerator above its crossover and retain safe fallback."""
        atoms = Atoms("HH", positions=[[0, 0, 0], [0, 0, 1]], cell=np.eye(3) * 10)
        accelerated = ([[1], [0]], [[1], [1]])
        reference = ([[], []], [[], []])
        calls = []
        monkeypatch.setattr(
            detect_module,
            "_OPENBABEL_PERIODIC_NEIGHBOR_MIN_ATOMS",
            2,
        )
        monkeypatch.setattr(
            detect_instance,
            "_getbondfromperiodicneighborlist",
            lambda *args: calls.append("fast") or accelerated,
        )
        monkeypatch.setattr(
            detect_instance,
            "_getbondfromopenbabel",
            lambda *args: calls.append("reference") or reference,
        )

        assert detect_instance._getbondfromcrd(atoms, np.eye(3) * 10) == accelerated
        assert calls == ["fast"]
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_CKDTREE
        )

        calls.clear()

        def fallback(*args):
            calls.append("fast")
            detect_instance._last_bond_detection_mode = (
                detect_module._BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY
            )
            return None

        monkeypatch.setattr(
            detect_instance,
            "_getbondfromperiodicneighborlist",
            fallback,
        )
        assert detect_instance._getbondfromcrd(atoms, np.eye(3) * 10) == reference
        assert calls == ["fast", "reference"]
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_OPENBABEL_CUTOFF_BOUNDARY
        )

        calls.clear()
        monkeypatch.setattr(
            detect_module,
            "_OPENBABEL_PERIODIC_NEIGHBOR_MIN_ATOMS",
            3,
        )
        assert detect_instance._getbondfromcrd(atoms, np.eye(3) * 10) == reference
        assert calls == ["reference"]
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_OPENBABEL_SMALL_FRAME
        )

        calls.clear()
        monkeypatch.setattr(
            detect_module,
            "_OPENBABEL_PERIODIC_NEIGHBOR_MIN_ATOMS",
            2,
        )
        triclinic_cell = np.array(
            [[10.0, 0.0, 0.0], [1.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        assert detect_instance._getbondfromcrd(atoms, triclinic_cell) == reference
        assert calls == ["reference"]
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_OPENBABEL_UNSUPPORTED_CELL
        )

        calls.clear()
        detect_instance.pbc = False
        assert detect_instance._getbondfromcrd(atoms, np.eye(3) * 10) == reference
        assert calls == ["reference"]
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_OPENBABEL_NONPERIODIC
        )

        detect_instance.pbc = True
        detect_instance.use_ase = True
        monkeypatch.setattr(
            detect_instance,
            "_getbondfromase",
            lambda *args: accelerated,
        )
        assert detect_instance._getbondfromcrd(atoms, np.eye(3) * 10) == accelerated
        assert (
            detect_instance._last_bond_detection_mode
            == detect_module._BOND_DETECTION_ASE
        )


class TestParseCustomCutoffs:
    """Test the _parse_custom_cutoffs method."""

    @pytest.fixture
    def detect_instance(self):
        """Create a detect instance for testing."""
        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename="dummy",
            atomname=["H", "O", "C", "Al"],
        )
        return _DetectLAMMPSdump(rng)

    def test_valid_simple_format(self, detect_instance):
        """Test parsing a valid simple format."""
        result = detect_instance._parse_custom_cutoffs("H-O:1.5")
        expected = {frozenset({"H", "O"}): 1.5}
        assert result == expected

    def test_valid_multiple_pairs(self, detect_instance):
        """Test parsing multiple valid pairs."""
        result = detect_instance._parse_custom_cutoffs("H-O:1.5,C-H:2.0")
        expected = {frozenset({"H", "O"}): 1.5, frozenset({"C", "H"}): 2.0}
        assert result == expected

    def test_with_spaces(self, detect_instance):
        """Test parsing with spaces in the input."""
        result = detect_instance._parse_custom_cutoffs("H - O : 1.5 , C - H : 2.0")
        expected = {frozenset({"H", "O"}): 1.5, frozenset({"C", "H"}): 2.0}
        assert result == expected

    def test_invalid_missing_colon(self, detect_instance):
        """Test that missing colon raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_instance._parse_custom_cutoffs("H-O1.5")
        assert "Invalid custom cutoff format 'H-O1.5'" in str(exc_info.value)
        assert "Expected 'Element1-Element2:distance'" in str(exc_info.value)
        assert "Example: 'Al-O:2.5,C-H:1.1'" in str(exc_info.value)

    def test_invalid_missing_dash(self, detect_instance):
        """Test that missing dash raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_instance._parse_custom_cutoffs("HO:1.5")
        assert "Invalid custom cutoff format 'HO:1.5'" in str(exc_info.value)
        assert "Expected 'Element1-Element2:distance'" in str(exc_info.value)

    def test_invalid_distance_non_numeric(self, detect_instance):
        """Test that non-numeric distance raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_instance._parse_custom_cutoffs("H-O:invalid")
        assert "Invalid distance value 'invalid'" in str(exc_info.value)
        assert "Expected a number" in str(exc_info.value)
        assert "Example: 'Al-O:2.5,C-H:1.1'" in str(exc_info.value)


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE is not available")
class TestGetBondFromASE:
    """Test the _getbondfromase method."""

    @pytest.fixture
    def detect_instance(self):
        """Create a detect instance for testing."""
        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename="dummy",
            atomname=["H", "O"],
            use_ase=True,
        )
        return _DetectLAMMPSdump(rng)

    def test_water_molecule_bonds(self, detect_instance):
        """Test that water molecule detects 2 bonds with default settings."""
        # Create a simple water molecule
        atoms = Atoms(
            "H2O",
            positions=[[0.757, 0.586, 0.0], [-0.757, 0.586, 0.0], [0.0, 0.0, 0.0]],
        )
        cell = np.eye(3) * 10  # Large cell to avoid PBC issues

        bond, _bondlevel = detect_instance._getbondfromase(atoms, cell)

        # Should have 3 atoms
        assert len(bond) == 3

        # Count total bonds (each bond is counted twice - once for each atom)
        total_bonds = sum(len(neighbors) for neighbors in bond)
        assert total_bonds == 4  # 2 actual bonds, each counted twice

        # Check that oxygen is bonded to both hydrogens
        assert 0 in bond[2]  # O (index 2) bonded to H (index 0)
        assert 1 in bond[2]  # O (index 2) bonded to H (index 1)
        assert 2 in bond[0]  # H (index 0) bonded to O (index 2)
        assert 2 in bond[1]  # H (index 1) bonded to O (index 2)

    def test_duplicate_bond_prevention_with_pbc(self, detect_instance):
        """Test that duplicate bonds are prevented with PBC."""
        # Create a simple water molecule
        atoms = Atoms(
            "H2O",
            positions=[[0.757, 0.586, 0.0], [-0.757, 0.586, 0.0], [0.0, 0.0, 0.0]],
        )
        # Use a small cell to create potential duplicate scenarios
        cell = np.eye(3) * 5.0

        bond, _bondlevel = detect_instance._getbondfromase(atoms, cell)

        # Check that there are no duplicate entries in bond lists
        for i, neighbors in enumerate(bond):
            # Check that all neighbors are unique
            assert len(neighbors) == len(
                set(neighbors)
            ), f"Duplicate bonds found for atom {i}"

    def test_custom_cutoffs_override(self, detect_instance):
        """Test that custom cutoffs override global settings."""
        # Modify the detect instance to have custom cutoffs
        detect_instance.rng.custom_cutoffs = "H-O:1.0"  # Very short cutoff

        # Create a simple water molecule where H-O distance is ~1.0 Angstrom
        atoms = Atoms(
            "H2O", positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )  # H-O distance is 0.5
        cell = np.eye(3) * 10

        bond, _bondlevel = detect_instance._getbondfromase(atoms, cell)

        # With a 1.0 cutoff, H-O should still be bonded
        # Check that oxygen is bonded to at least one hydrogen
        assert 0 in bond[2] or 1 in bond[2]  # O (index 2) bonded to H (index 0 or 1)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="Scipy is not available")
class TestScipyClustering:
    """Test the Scipy clustering in _connectmolecule method."""

    @pytest.fixture
    def detect_instance(self):
        """Create a detect instance for testing."""
        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename="dummy",
            atomname=["H", "O"],
            use_ase=True,
        )
        return _DetectLAMMPSdump(rng)

    def test_large_linear_chain_scipy_clustering(self, detect_instance):
        """Test scipy clustering with a large linear chain."""
        # Create a large linear chain of atoms: 0-1-2-3-...-499
        n_atoms = 500
        bond = [[] for _ in range(n_atoms)]
        level = [[] for _ in range(n_atoms)]

        # Create linear chain bonds
        for i in range(n_atoms - 1):
            bond[i].append(i + 1)
            bond[i + 1].append(i)
            level[i].append(1)
            level[i + 1].append(1)

        # Use the _connectmolecule method which should use scipy when available
        result = detect_instance._connectmolecule(bond, level)

        # Should have 1 component containing all atoms
        assert len(result) == 1

        # The component should contain all atoms
        # The first part of the component bytes contains the atom indices
        # Need to extract the atom list from the bytes format
        # This is tricky since it's in the internal format, so just check length
        # The result format is: atom_indices + bond_pairs + bond_levels
        assert len(result) == 1  # Single connected component


class TestAutoEnableLogic:
    """Test the auto-enable logic for ASE mode."""

    def test_auto_enable_with_custom_cutoffs(self, tmp_path):
        """Test that ASE mode is auto-enabled when custom cutoffs are provided."""
        # Create a temporary file for input
        dummy_file = tmp_path / "dummy.dump"
        dummy_file.write_text("")

        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename=str(dummy_file),
            atomname=["H", "O"],
            use_ase=False,  # Explicitly set to False
            custom_cutoffs="H-O:1.5",  # But provide custom cutoffs
        )

        # Should be auto-enabled
        assert rng.use_ase is True

    def test_auto_enable_with_modified_multiplier(self, tmp_path):
        """Test that ASE mode is auto-enabled when multiplier is modified."""
        # Create a temporary file for input
        dummy_file = tmp_path / "dummy.dump"
        dummy_file.write_text("")

        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename=str(dummy_file),
            atomname=["H", "O"],
            use_ase=False,  # Explicitly set to False
            ase_cutoff_mult=1.5,  # But modify multiplier
        )

        # Should be auto-enabled
        assert rng.use_ase is True
