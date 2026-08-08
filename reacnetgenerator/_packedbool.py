# SPDX-License-Identifier: LGPL-3.0-or-later
"""Disk-backed packed boolean matrices for bounded temporary state."""

import operator

import numpy as np

_PACKED_BOOL_SPARSE_SELECTION_DIVISOR = 8


def _is_full_slice(index):
    return (
        isinstance(index, slice)
        and index.start is None
        and index.stop is None
        and index.step is None
    )


class _PackedBoolColumn:
    """Lazy logical column that decodes only requested row blocks."""

    def __init__(self, matrix, column):
        self._matrix = matrix
        self._column = operator.index(column)
        self.shape = (matrix.shape[0],)
        self.dtype = np.dtype(np.bool_)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        return self._matrix._read_column(self._column, rows)

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self[:])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = values.copy()
        return values


class _PackedBoolMatrix:
    """Store a logical row-major boolean matrix using one bit per cell."""

    def __init__(self, path, shape, *, mode):
        self.path = path
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("Packed boolean matrix shape must be two positive values")
        self.dtype = np.dtype(np.bool_)
        self.packed_shape = (self.shape[0], (self.shape[1] + 7) // 8)
        self.data = np.memmap(
            path,
            mode=mode,
            dtype=np.uint8,
            shape=self.packed_shape,
        )

    @property
    def nbytes(self):
        return int(np.prod(self.packed_shape, dtype=np.int64))

    def __len__(self):
        return self.shape[0]

    def _read_column(self, column, rows):
        column = operator.index(column)
        if column < 0:
            column += self.shape[1]
        if column < 0 or column >= self.shape[1]:
            raise IndexError("Packed boolean matrix column is out of range")
        values = self.data[rows, column // 8]
        return np.not_equal(
            np.bitwise_and(values, np.uint8(1 << (column % 8))),
            0,
        )

    def _unpack_rows(self, rows):
        packed = np.asarray(self.data[rows])
        values = np.unpackbits(packed, axis=-1, bitorder="little")
        return values[..., : self.shape[1]].view(np.bool_)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("Packed boolean matrix requires two indices")
            rows, columns = index
            try:
                column = operator.index(columns)
            except TypeError:
                values = self._unpack_rows(rows)
                return values[..., columns]
            lazy_column = _PackedBoolColumn(self, column)
            if _is_full_slice(rows):
                return lazy_column
            return lazy_column[rows]
        return self._unpack_rows(index)

    def __setitem__(self, index, value):
        if not _is_full_slice(index):
            raise TypeError("Packed boolean matrix only supports whole-matrix fill")
        if not np.isscalar(value):
            raise TypeError("Packed boolean matrix fill value must be a scalar")
        self.data.fill(255 if bool(value) else 0)
        if value and self.shape[1] % 8:
            self.data[:, -1] &= np.uint8((1 << (self.shape[1] % 8)) - 1)

    def __array__(self, dtype=None, copy=None):
        values = self._unpack_rows(slice(None))
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if copy:
            values = values.copy()
        return values

    def _mark_indexed(self, atoms, frames, overlap):
        frames = np.asarray(frames, dtype=np.int64).reshape((-1,))
        if np.any(frames < 0):
            frames = frames.copy()
            frames[frames < 0] += self.shape[1]
        if np.any((frames < 0) | (frames >= self.shape[1])):
            raise IndexError("Packed boolean matrix frame is out of range")
        if len(atoms) == 0 or len(frames) == 0:
            return
        active_count = int(np.count_nonzero(overlap))
        if active_count == 0:
            return
        if active_count * _PACKED_BOOL_SPARSE_SELECTION_DIVISOR <= overlap.size:
            active_positions = np.flatnonzero(overlap)
            atom_positions, frame_positions = np.divmod(
                active_positions,
                len(frames),
            )
            selected_frames = frames[frame_positions]
            np.bitwise_or.at(
                self.data,
                (atoms[atom_positions], selected_frames // 8),
                np.asarray(1 << (selected_frames % 8), dtype=np.uint8),
            )
            return
        bit_masks = np.asarray(1 << (frames % 8), dtype=np.uint8)
        write_masks = overlap.view(np.uint8).copy()
        np.multiply(write_masks, bit_masks, out=write_masks)
        np.bitwise_or.at(
            self.data,
            (atoms[:, np.newaxis], (frames // 8)[np.newaxis, :]),
            write_masks,
        )

    def _mark_contiguous(self, atoms, frame_start, overlap):
        frame_count = overlap.shape[1]
        prefix_count = min(frame_count, (-frame_start) % 8)
        if prefix_count:
            self._mark_indexed(
                atoms,
                np.arange(frame_start, frame_start + prefix_count),
                overlap[:, :prefix_count],
            )
        remaining_count = frame_count - prefix_count
        middle_count = remaining_count - remaining_count % 8
        if middle_count:
            middle_start = prefix_count
            middle_stop = middle_start + middle_count
            byte_start = (frame_start + middle_start) // 8
            byte_stop = byte_start + middle_count // 8
            packed = np.packbits(
                overlap[:, middle_start:middle_stop],
                axis=1,
                bitorder="little",
            )
            matrix_index = (atoms, slice(byte_start, byte_stop))
            selected = self.data[matrix_index]
            np.bitwise_or(selected, packed, out=selected)
            self.data[matrix_index] = selected
        suffix_start = prefix_count + middle_count
        if suffix_start < frame_count:
            self._mark_indexed(
                atoms,
                np.arange(
                    frame_start + suffix_start,
                    frame_start + frame_count,
                ),
                overlap[:, suffix_start:],
            )

    def mark(self, atoms, frames, overlap):
        """Set logical cells selected by an already bounded overlap mask."""
        atoms = np.asarray(atoms, dtype=np.int64).reshape((-1,))
        overlap = np.asarray(overlap, dtype=np.bool_)
        if overlap.ndim != 2 or overlap.shape[0] != len(atoms):
            raise ValueError("Conflict overlap shape does not match selected atoms")
        if isinstance(frames, slice):
            frame_start, frame_stop, frame_step = frames.indices(self.shape[1])
            if frame_step != 1:
                raise ValueError("Packed contiguous conflict writes require step=1")
            if overlap.shape[1] != frame_stop - frame_start:
                raise ValueError("Conflict overlap shape does not match frame slice")
            self._mark_contiguous(atoms, frame_start, overlap)
            return
        frame_values = np.asarray(frames)
        if frame_values.ndim != 1:
            frame_values = frame_values.reshape((-1,))
        if overlap.shape[1] != len(frame_values):
            raise ValueError("Conflict overlap shape does not match selected frames")
        self._mark_indexed(atoms, frame_values, overlap)

    def flush(self):
        if self.data is not None:
            self.data.flush()

    def close(self):
        data = getattr(self, "data", None)
        if data is None:
            return
        data.flush()
        mmap = getattr(data, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.data = None

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass
