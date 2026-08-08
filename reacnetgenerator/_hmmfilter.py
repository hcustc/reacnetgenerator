# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""HMM Filter.

In order to filter noise, a two-state HMM was adopted, which can be
described as a transition matrix A, an emission matrix B, and an initial
state vector π. The existence of molecules can be converted into 0-1 signals.
In order to predict the state sequence according to the output sequence,
Viterbi Algorithm is used to acquire the path with the most likely hidden
state sequence V called Viterbi path.

References
----------
.. [1] Wang, L.-P.; McGibbon, R. T.; Pande, V. S.; Martínez, T. J. Automated
   discovery and refinement of reactive molecular dynamics pathways. J. Chem.
   Theory Comput. 2016, 12(2), 638-649.
.. [2] Rabiner, L. R. A trtorial on hidden Markov models and selected
   applications in speech recognition. Proc. IEEE 1989, 77(2), 257-286.
.. [3] Forney, G. D. The viterbi algorithm. Porc. IEEE 1973, 61(3), 268-278.
"""

import itertools
import os
import tempfile
from contextlib import ExitStack
from typing import NamedTuple

import numpy as np
from tqdm.auto import tqdm

try:
    # hmmlearn v0.2.8 renamed MultinomialHMM to CategoricalHMM
    from hmmlearn.hmm import CategoricalHMM as MultinomialHMM
except ImportError:
    from hmmlearn.hmm import MultinomialHMM

from ._logging import logger
from .utils import (
    SharedRNGData,
    WriteBuffer,
    _frame_indices_fit_signal_memory,
    appendIfNotNone,
    bytestolist,
    check_zero_signal,
    idx_to_signal,
    listtobytes,
    read_compressed_block,
    run_mp,
)

_HMM_TARGET_SIGNAL_VALUES_PER_CHUNK = 1_000_000
_HMM_MAX_CHUNKSIZE = 100
_HMM_DEFAULT_INFLIGHT_RECORDS_PER_WORKER = 150
_HMM_TARGET_INFLIGHT_CHUNKS_PER_WORKER = 2


class _SmilesWorkMetrics(NamedTuple):
    """Compressed structure work observed for retained molecule records."""

    total_compressed_bytes: int
    max_compressed_record_bytes: int


def _hmm_parallel_pool_limits(nproc: int, frame_count: int) -> tuple[int, int]:
    """Bound decoded signal work while preserving batching for short trajectories."""
    workers = max(1, int(nproc))
    frames = max(1, int(frame_count))
    chunksize = max(
        1,
        min(
            _HMM_MAX_CHUNKSIZE,
            _HMM_TARGET_SIGNAL_VALUES_PER_CHUNK // frames,
        ),
    )
    default_max_inflight = workers * _HMM_DEFAULT_INFLIGHT_RECORDS_PER_WORKER
    working_max_inflight = workers * _HMM_TARGET_INFLIGHT_CHUNKS_PER_WORKER * chunksize
    max_inflight = min(
        default_max_inflight,
        max(chunksize, working_max_inflight),
    )
    return chunksize, max_inflight


class _HMMFilter(SharedRNGData):
    runHMM: bool
    getoriginfile: bool
    printfiltersignal: bool
    moleculetempfilename: str
    nproc: int
    temp1it: int
    p: np.ndarray
    a: np.ndarray
    b: np.ndarray
    step: int
    smilesworkmetrics: _SmilesWorkMetrics

    def __init__(self, rng):
        SharedRNGData.__init__(
            self,
            rng,
            [
                "runHMM",
                "getoriginfile",
                "printfiltersignal",
                "moleculetempfilename",
                "nproc",
                "temp1it",
                "p",
                "a",
                "b",
                "step",
            ],
            [
                "moleculetemp2filename",
                "originfilename",
                "hmmfilename",
                "hmmit",
                "smilesworkmetrics",
            ],
        )

    def filter(self):
        """HMM Filters.

        Timesteps of molecules are converted to a visible output sequence.
        O^m=(o_t^m) is given by o_t^m={1, if m exists; 0, otherwise}.
        Similarly, a hidden state sequence I^m=(i_t^m) is given by
        i_t^m={1, if m exists; 0, otherwise.}
        """
        if self.runHMM:
            self._initHMM()
        self._calhmm()
        self.returnkeys()

    def _initHMM(self):
        self._model = MultinomialHMM(n_components=2, algorithm="viterbi")
        self._model.startprob_ = self.p
        self._model.transmat_ = self.a
        self._model.emissionprob_ = self.b

    def _getoriginandhmm(self, item):
        line_c = item
        if not self.runHMM and _frame_indices_fit_signal_memory(
            line_c[-1],
            self.step,
        ):
            return None, None, line_c
        value = bytestolist(line_c[-1])
        origin = idx_to_signal(value, self.step)
        originbytes = (
            listtobytes(origin) if self.getoriginfile or not self.runHMM else None
        )
        hmmbytes = None
        if self.runHMM:
            hmmsignal = self._model.predict(origin).astype(bool)
            if check_zero_signal(hmmsignal) or self.printfiltersignal:
                hmmbytes = listtobytes(hmmsignal)
        return originbytes, hmmbytes, line_c

    def _iter_filter_results(self, blocks):
        """Bypass multiprocessing until no-HMM work needs signal expansion."""
        chunksize, max_inflight = _hmm_parallel_pool_limits(self.nproc, self.step)
        if self.runHMM:
            logger.info(
                "HMM filter parallel limits: %d frames, chunksize=%d, "
                "max_inflight=%d",
                self.step,
                chunksize,
                max_inflight,
            )
            yield from run_mp(
                self.nproc,
                func=self._getoriginandhmm,
                l=blocks,
                nlines=4,
                total=self.temp1it,
                desc="HMM filter",
                unit="molecule",
                chunksize=chunksize,
                max_inflight=max_inflight,
            )
            return

        records = itertools.zip_longest(*[blocks] * 4)
        processed = 0
        self._no_hmm_parallel_fallback = False
        with tqdm(
            total=self.temp1it,
            desc="HMM filter",
            unit="molecule",
            disable=None,
        ) as progress:
            for record in records:
                if _frame_indices_fit_signal_memory(record[-1], self.step):
                    result = self._getoriginandhmm(record)
                    processed += 1
                    progress.update()
                    yield result
                    continue

                self._no_hmm_parallel_fallback = True
                logger.info(
                    "No-HMM dense fallback parallel limits: %d frames, "
                    "chunksize=%d, max_inflight=%d",
                    self.step,
                    chunksize,
                    max_inflight,
                )
                remaining_records = itertools.chain((record,), records)
                for result in run_mp(
                    self.nproc,
                    func=self._getoriginandhmm,
                    l=remaining_records,
                    total=max(0, self.temp1it - processed),
                    desc=None,
                    unit="molecule",
                    bar=False,
                    chunksize=chunksize,
                    max_inflight=max_inflight,
                    unordered=False,
                ):
                    progress.update()
                    yield result
                return

    def _calhmm(self):
        with (
            (
                WriteBuffer(tempfile.NamedTemporaryFile("wb", delete=False))
                if self.getoriginfile or not self.runHMM
                else ExitStack()
            ) as fo,
            (
                WriteBuffer(tempfile.NamedTemporaryFile("wb", delete=False))
                if self.runHMM
                else ExitStack()
            ) as fh,
            open(self.moleculetempfilename, "rb") as ft,
            (
                WriteBuffer(tempfile.NamedTemporaryFile("wb", delete=False))
                if self.runHMM
                else ExitStack()
            ) as ft2,
        ):
            if self.runHMM:
                assert not isinstance(ft2, ExitStack)
                self.moleculetemp2filename = ft2.name
            else:
                self.moleculetemp2filename = self.moleculetempfilename
            if self.getoriginfile or not self.runHMM:
                assert not isinstance(fo, ExitStack)
                self.originfilename = fo.name
            else:
                self.originfilename = None
            if self.runHMM:
                assert not isinstance(fh, ExitStack)
                self.hmmfilename = fh.name
            else:
                self.hmmfilename = None
            results = self._iter_filter_results(read_compressed_block(ft))
            hmmit = 0
            total_compressed_bytes = 0
            max_compressed_record_bytes = 0
            direct_origin_molecules = 0
            origin_output_bytes = 0
            for originbytes, hmmbytes, line_c in results:
                if not self.runHMM or originbytes is not None or hmmbytes is not None:
                    appendIfNotNone(fo, originbytes)
                    appendIfNotNone(fh, hmmbytes)
                    hmmit += 1
                    if originbytes is not None:
                        origin_output_bytes += len(originbytes)
                    if not self.runHMM and _frame_indices_fit_signal_memory(
                        line_c[-1],
                        self.step,
                    ):
                        direct_origin_molecules += 1
                    structure_bytes = len(line_c[0]) + len(line_c[1]) + len(line_c[2])
                    total_compressed_bytes += structure_bytes
                    max_compressed_record_bytes = max(
                        max_compressed_record_bytes,
                        structure_bytes,
                    )
                    if self.runHMM:
                        assert not isinstance(ft2, ExitStack)
                        ft2.extend(line_c)
        self.hmmit = hmmit
        self.smilesworkmetrics = _SmilesWorkMetrics(
            total_compressed_bytes=total_compressed_bytes,
            max_compressed_record_bytes=max_compressed_record_bytes,
        )
        if not self.runHMM:
            logger.info(
                "No-HMM filter passthrough: %d/%d direct molecules, %.3f MiB "
                "fallback origins, %.3f MiB molecule copy avoided, parallel "
                "fallback used=%s",
                direct_origin_molecules,
                hmmit,
                origin_output_bytes / (1024 * 1024),
                os.path.getsize(self.moleculetempfilename) / (1024 * 1024),
                self._no_hmm_parallel_fallback,
            )
