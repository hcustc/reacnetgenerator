# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
"""Test ReacNetGen."""

import fileinput
import itertools
import json
import logging
import os
from collections import Counter
from tkinter import END, TclError
from types import SimpleNamespace

import numpy as np
import pytest

import reacnetgenerator._path as path_module
import reacnetgenerator.reacnetgen as reacnetgen_module
from reacnetgenerator import ReacNetGenerator
from reacnetgenerator._detect import _Detect
from reacnetgenerator._hmmfilter import _HMMFilter
from reacnetgenerator._path import _CollectSMILESPaths
from reacnetgenerator._reaction import ReactionsFinder
from reacnetgenerator.commandline import parm2cmd
from reacnetgenerator.gui import GUI
from reacnetgenerator.utils import (
    checksha256,
    download_multifiles,
    get_timestep_value,
    listtobytes,
)

with open(os.path.join(os.path.dirname(__file__), "test.json")) as f:
    test_data = json.load(f)


def _range_pairs(collector, frames):
    return [
        (int(start), int(end))
        for starts, ends in collector._itermoleculeranges(frames)
        for start, end in zip(starts, ends)
    ]


def test_cpu_allocation_log_exposes_slurm_affinity(caplog, monkeypatch, tmp_path):
    """Startup diagnostics should expose a Slurm/affinity mismatch."""
    monkeypatch.setattr(
        reacnetgen_module.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 4, 5},
        raising=False,
    )
    monkeypatch.setattr(reacnetgen_module.os, "cpu_count", lambda: 10)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    caplog.set_level(logging.INFO, logger="reacnetgenerator._logging")

    ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="lammpsbondfile",
        atomname=["H"],
        nproc=64,
    )

    assert (
        "CPU allocation: nproc=64; available=4; logical=10; "
        "affinity=0-1,4-5; SLURM_CPUS_PER_TASK=64"
    ) in caplog.text
    assert "Requested nproc=64 exceeds 4 CPUs visible to this process" in caplog.text
    assert "SLURM_CPUS_PER_TASK=64 but process affinity exposes 4 CPUs" in caplog.text


def test_cpu_allocation_log_without_affinity(caplog, monkeypatch, tmp_path):
    """Platforms without affinity support should report the fallback capacity."""
    monkeypatch.delattr(reacnetgen_module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(reacnetgen_module.os, "cpu_count", lambda: 10)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    caplog.set_level(logging.INFO, logger="reacnetgenerator._logging")

    rng = ReacNetGenerator(
        inputfilename=str(tmp_path / "dummy"),
        inputfiletype="lammpsbondfile",
        atomname=["H"],
    )

    assert rng.nproc == 10
    assert (
        "CPU allocation: nproc=10; available=10; logical=10; "
        "affinity=unavailable; SLURM_CPUS_PER_TASK=unset"
    ) in caplog.text


class TestReacNetGen:
    """Test ReacNetGenerator."""

    @pytest.fixture(autouse=True)
    def chdir(self, tmp_path):
        """Change directory to tmp_path."""
        start_direcroty = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(start_direcroty)

    @pytest.fixture(
        params=[
            pytest.param(
                param, marks=(pytest.mark.xfail if param.get("xfail", False) else ())
            )
            for param in test_data
        ]
    )
    def reacnetgen_param(self, request):
        """Fixture for ReacNetGenerator parameters."""
        return request.param

    @pytest.fixture()
    def reacnetgen(self, reacnetgen_param):
        """Fixture for ReacNetGenerator."""
        rngclass = ReacNetGenerator(**reacnetgen_param["rngparams"])
        yield rngclass
        assert checksha256(
            rngclass.reactionfilename, reacnetgen_param.get("reaction_sha256", [])
        )
        assert os.path.exists(rngclass.reactionfilename)
        assert os.path.exists(rngclass.resultfilename)

    def test_reacnetgen(self, reacnetgen):
        """Test main process of ReacNetGen."""
        reacnetgen.runanddraw()

    def test_reacnetgen_step(self, reacnetgen):
        """Test main process of ReacNetGen."""
        reacnetgen.run()
        reacnetgen.draw()
        reacnetgen.report()

    @pytest.fixture()
    def reacnetgengui(self):
        """Fixture for GUI test."""
        try:
            gui = GUI()
            yield gui
            try:
                gui.root.destroy()
            except TclError:
                pass
        except TclError:
            pytest.skip("No display for GUI.")

    def test_gui(self, reacnetgengui):
        """Test GUI of ReacNetGen."""
        reacnetgengui.root.after(100, reacnetgengui.root.destroy)
        reacnetgengui.gui()

    def test_gui_openandrun(self, reacnetgengui, mocker, reacnetgen_param):
        """Test GUI of ReacNetGenerator."""
        pp = reacnetgen_param["rngparams"]
        mocker.patch(
            "tkinter.filedialog.askopenfilename", return_value=pp["inputfilename"]
        )
        mocker.patch("tkinter.messagebox.showerror")
        download_multifiles(pp.get("urls", []))
        reacnetgengui._atomnameet.delete(0, END)
        reacnetgengui._atomnameet.insert(0, " ".join(pp["atomname"]))
        if pp["inputfiletype"] in ["lammpsbondfile", "lammpsdumpfile"]:
            reacnetgengui._filetype.set(pp["inputfiletype"])
        reacnetgengui._openbtn.invoke()
        reacnetgengui._runbtn.invoke()

    def test_commandline_help(self, script_runner):
        """Test commandline of ReacNetGenerator."""
        ret = script_runner.run(["reacnetgenerator", "-h"])
        assert ret.success

    def test_commandline_version(self, script_runner):
        """Test commandline of ReacNetGenerator."""
        ret = script_runner.run(["reacnetgenerator", "--version"])
        assert ret.success

    def test_commandline_run(self, script_runner, reacnetgen_param):
        """Test commandline of ReacNetGenerator."""
        ret = script_runner.run(parm2cmd(reacnetgen_param["rngparams"]))
        assert ret.success

    def test_benchmark_detect(self, benchmark, reacnetgen_param):
        """Benchmark _readstepfunc."""
        reacnetgen = ReacNetGenerator(**reacnetgen_param["rngparams"])
        reacnetgen._process((reacnetgen.Status.DOWNLOAD,))
        detectclass = _Detect.gettype(reacnetgen)
        with fileinput.input(files=detectclass.inputfilename) as f:
            nlines = detectclass._readNfunc(f)
        with fileinput.input(files=detectclass.inputfilename) as f:
            lines = next(itertools.zip_longest(*[f] * nlines))

        @benchmark
        def bench():
            detectclass._readstepfunc((0, lines))

    def test_benchmark_hmm(self, benchmark, reacnetgen_param):
        """Benchmark _getoriginandhmm."""
        reacnetgen = ReacNetGenerator(**reacnetgen_param["rngparams"])
        reacnetgen.step = 250000
        hmmclass = _HMMFilter(reacnetgen)
        if hmmclass.runHMM:
            hmmclass._initHMM()
        rng = np.random.default_rng()
        index = np.sort(rng.choice(hmmclass.step, hmmclass.step // 2, replace=False))
        compressed_bytes = [
            listtobytes((5, 6)),
            listtobytes(((5, 6, 1),)),
            listtobytes(index),
        ]

        @benchmark
        def bench():
            hmmclass._getoriginandhmm(compressed_bytes)

    def test_re(self, reacnetgen_param):
        """Test regular expression of _HTMLResult."""
        reacnetgen = ReacNetGenerator(**reacnetgen_param["rngparams"])
        r = _CollectSMILESPaths(reacnetgen)
        r.atomname = np.array(["C", "H", "O", "Na", "Cl"])
        assert r._re("C") == "[C]"
        assert r._re("[C]") == "[C]"
        assert r._re("[CH]") == "[CH]"
        assert r._re("Na") == "[Na]"
        assert r._re("[H]c(Cl)C([H])Cl") == "[H][c]([Cl])[C]([H])[Cl]"

    def test_re_mo(self, reacnetgen_param):
        """Test regular expression of _HTMLResult."""
        reacnetgen = ReacNetGenerator(**reacnetgen_param["rngparams"])
        r = _CollectSMILESPaths(reacnetgen)
        r.atomname = np.array(["Mo", "O"])
        assert r._re("[Mo]") == "[Mo]"

    def test_reaction_event_details(self, tmp_path):
        """Single reaction events should expose normalized reaction pairs."""
        finder = ReactionsFinder(
            SimpleNamespace(
                step=2,
                mname=np.array(["A", "B", "C"]),
                reactionabcdfilename=str(tmp_path / "out.reactionabcd"),
                printreactionevent=True,
                nproc=1,
            )
        )
        item = listtobytes(
            (
                0,
                np.array([1, 1, 2, 2]),
                np.array([3, 3, 3, 3]),
                np.zeros(4, dtype=int),
                np.zeros(4, dtype=int),
            )
        )

        assert finder._getstepreaction(item) == (0, Counter({("A+B", "C"): 1}))

    def test_reaction_event_default_is_off(self, tmp_path):
        """Reaction event details should not be calculated unless requested."""
        finder = ReactionsFinder(
            SimpleNamespace(
                step=2,
                mname=np.array(["A", "B", "C"]),
                reactionabcdfilename=str(tmp_path / "out.reactionabcd"),
                printreactionevent=False,
                nproc=1,
            )
        )
        item = listtobytes(
            (
                np.array([1, 1, 2, 2]),
                np.array([3, 3, 3, 3]),
                np.zeros(4, dtype=int),
                np.zeros(4, dtype=int),
            )
        )

        assert finder._getstepreaction(item) == ["A+B->C"]

    def test_get_timestep_value(self):
        """Stored timestep metadata should normalize to the timestep value."""
        assert get_timestep_value((0, 100)) == 100
        assert get_timestep_value(np.int64(100)) == 100
        assert get_timestep_value(100) == 100

    def test_molecule_time_formatting(self, tmp_path):
        """Molecule timeline ranges should be optional and filterable."""
        reacnetgen = ReacNetGenerator(
            inputfilename=str(tmp_path / "dummy"),
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            printmoleculetime=True,
            moleculeframes=[2],
        )
        collector = _CollectSMILESPaths(reacnetgen)
        collector.timestep = {0: 100, 1: 200, 2: 300}

        frames = np.array([0, 2])
        timesteps = collector._getmoleculetimesteps(frames)

        assert timesteps == [100, 300]
        assert reacnetgen.timedoutputfilename == str(tmp_path / "dummy.timeline.h5")
        assert collector._formatmoleculename("C", np.array([0, 1]), [[0, 1, 1]]) == (
            "C 0;1 0,1,1"
        )
        assert _range_pairs(collector, frames) == [(2, 2)]

    def test_molecule_time_filter_by_timestep(self):
        """Molecule timeline filtering should accept original timestep values."""
        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            moleculetimesteps=[300],
        )
        collector = _CollectSMILESPaths(reacnetgen)
        collector.timestep = {0: 100, 2: 300}

        assert reacnetgen.printmoleculetime is True
        assert _range_pairs(collector, [0, 2]) == [(2, 2)]

    def test_unfiltered_molecule_ranges_do_not_read_timesteps(self):
        """Compress frame ranges without per-frame timestep dictionary lookups."""

        class NoTimestepLookup(dict):
            def __getitem__(self, key):
                raise AssertionError(f"unexpected timestep lookup for frame {key}")

        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            printmoleculetime=True,
        )
        collector = _CollectSMILESPaths(reacnetgen)
        collector.timestep = NoTimestepLookup()

        assert _range_pairs(collector, [0, 1, 1, 3, 4, 7]) == [
            (0, 1),
            (3, 4),
            (7, 7),
        ]

    def test_fragmented_molecule_ranges_are_emitted_as_numpy_blocks(self):
        """Keep highly fragmented range generation off the Python object path."""
        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            printmoleculetime=True,
        )
        collector = _CollectSMILESPaths(reacnetgen)
        frames = np.arange(0, 8194, 2, dtype=np.uint64)

        blocks = list(collector._itermoleculeranges(frames))

        assert [len(starts) for starts, _ in blocks] == [4096, 1]
        assert all(
            isinstance(values, np.ndarray) for block in blocks for values in block
        )
        np.testing.assert_array_equal(
            np.concatenate([starts for starts, _ in blocks]),
            frames,
        )
        np.testing.assert_array_equal(
            np.concatenate([ends for _, ends in blocks]),
            frames,
        )

    def test_molecule_ranges_join_across_scan_blocks(self, monkeypatch):
        """Preserve duplicate/contiguous semantics at scan block boundaries."""
        monkeypatch.setattr(path_module, "_RANGE_SCAN_ROWS", 4)
        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            printmoleculetime=True,
        )
        collector = _CollectSMILESPaths(reacnetgen)

        assert _range_pairs(collector, [0, 1, 1, 2, 4, 5, 8]) == [
            (0, 2),
            (4, 5),
            (8, 8),
        ]

    def test_molecule_time_filters_match_same_occurrence(self):
        """Combined frame and timestep filters should match the same timeline row."""
        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            moleculeframes=[0],
            moleculetimesteps=[300],
        )
        collector = _CollectSMILESPaths(reacnetgen)
        collector.timestep = {0: 100, 2: 300}

        assert _range_pairs(collector, [0, 2]) == []

        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            moleculeframes=[2],
            moleculetimesteps=[300],
        )
        collector = _CollectSMILESPaths(reacnetgen)
        collector.timestep = {0: 100, 2: 300}

        assert _range_pairs(collector, [0, 2]) == [(2, 2)]

    def test_empty_molecule_filters_are_ignored(self, tmp_path):
        """Empty frame and timestep filters should behave like omitted filters."""
        reacnetgen = ReacNetGenerator(
            inputfilename=str(tmp_path / "dummy"),
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            moleculeframes=[],
            moleculetimesteps=[],
        )
        assert reacnetgen.moleculeframes is None
        assert reacnetgen.moleculetimesteps is None
        assert reacnetgen.printmoleculetime is False
        assert not (tmp_path / "dummy.timeline.h5").exists()

    def test_molecule_filter_normalization_accepts_sequences(self):
        """Tuple and array molecule filters should normalize to integer lists."""
        reacnetgen = ReacNetGenerator(
            inputfilename="dummy",
            inputfiletype="lammpsbondfile",
            atomname=["H", "O"],
            moleculeframes=(1, 2),
            moleculetimesteps=np.array([100, 300]),
        )

        assert reacnetgen.moleculeframes == [1, 2]
        assert reacnetgen.moleculetimesteps == [100, 300]

    def test_parm2cmd_preserves_zero_molecule_filters(self):
        """Frame and timestep 0 are valid molecule filters."""
        cmd = parm2cmd(
            {
                "inputfilename": "dummy",
                "inputfiletype": "lammpsbondfile",
                "atomname": ["H", "O"],
                "moleculeframes": 0,
                "moleculetimesteps": 0,
            }
        )

        assert "--molecule-frame" in cmd
        assert cmd[cmd.index("--molecule-frame") + 1] == "0"
        assert "--molecule-timestep" in cmd
        assert cmd[cmd.index("--molecule-timestep") + 1] == "0"

    def test_parm2cmd_omits_empty_molecule_filters(self):
        """Empty molecule filters should not emit value-requiring CLI options."""
        cmd = parm2cmd(
            {
                "inputfilename": "dummy",
                "inputfiletype": "lammpsbondfile",
                "atomname": ["H", "O"],
                "moleculeframes": [],
                "moleculetimesteps": [],
            }
        )

        assert "--molecule-frame" not in cmd
        assert "--molecule-timestep" not in cmd


# Additional test for the auto-enable logic
class TestAutoEnableLogic:
    """Test the auto-enable logic for ASE mode."""

    @pytest.fixture(autouse=True)
    def chdir(self, tmp_path):
        """Change directory to tmp_path."""
        start_directory = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(start_directory)

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

    def test_no_auto_enable_with_defaults(self, tmp_path):
        """Test that ASE mode is not auto-enabled when using defaults."""
        # Create a temporary file for input
        dummy_file = tmp_path / "dummy.dump"
        dummy_file.write_text("")

        rng = ReacNetGenerator(
            inputfiletype="lammpsdumpfile",
            inputfilename=str(dummy_file),
            atomname=["H", "O"],
            use_ase=False,  # Explicitly set to False
            ase_cutoff_mult=1.2,  # Default value
            custom_cutoffs=None,  # Default value
        )

        # Should remain False
        assert rng.use_ase is False
