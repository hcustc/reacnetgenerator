# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact molecule-to-species name mappings used by PATH analysis."""

from __future__ import annotations

import numpy as np


def _unsigned_dtype_for_maximum(maximum_value: int) -> np.dtype:
    """Return the narrowest unsigned dtype that contains ``maximum_value``."""
    maximum = max(0, int(maximum_value))
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise OverflowError("Value exceeds unsigned 64-bit integer storage")


class _MoleculeNameTable:
    """Map every molecule ID to one deduplicated species-name entry."""

    def __init__(self, ids, names, *, validate: bool = True) -> None:
        self.ids = np.asarray(ids)
        self.names = np.asarray(names)
        if self.ids.ndim != 1 or self.names.ndim != 1:
            raise ValueError("Molecule name IDs and values must be one-dimensional")
        if not np.issubdtype(self.ids.dtype, np.unsignedinteger):
            raise TypeError("Molecule name IDs must use an unsigned integer dtype")
        if validate and self.ids.size and int(self.ids.max()) >= len(self.names):
            raise ValueError("Molecule name ID is outside the species-name table")

    @classmethod
    def from_names(cls, values):
        values = list(values)
        builder = _MoleculeNameBuilder(len(values))
        for value in values:
            builder.append(value)
        return builder.finish()

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return (self.names[int(name_id)] for name_id in self.ids)

    def __getitem__(self, index):
        return self.names[self.ids[index]]

    def __array__(self, dtype=None, copy=None):
        values = self.names[self.ids]
        if copy is None:
            return np.asarray(values, dtype=dtype)
        return np.array(values, dtype=dtype, copy=copy)


class _MoleculeNameBuilder:
    """Build a fixed-capacity compact name table without retaining all strings."""

    def __init__(self, molecule_count: int) -> None:
        self.ids = np.empty(
            int(molecule_count),
            dtype=_unsigned_dtype_for_maximum(max(0, int(molecule_count) - 1)),
        )
        self.names = []
        self._name_ids = {}
        self._count = 0

    def append(self, name: str) -> None:
        if self._count >= len(self.ids):
            raise RuntimeError("More molecule names than the declared count")
        name = str(name)
        name_id = self._name_ids.get(name)
        if name_id is None:
            name_id = len(self.names)
            self._name_ids[name] = name_id
            self.names.append(name)
        self.ids[self._count] = name_id
        self._count += 1

    def finish(self) -> _MoleculeNameTable:
        if self._count != len(self.ids):
            raise RuntimeError("Fewer molecule names than the declared count")
        names = np.asarray(self.names, dtype=str)
        target_dtype = _unsigned_dtype_for_maximum(max(0, len(names) - 1))
        ids = (
            self.ids
            if self.ids.dtype == target_dtype
            else self.ids.astype(target_dtype)
        )
        return _MoleculeNameTable(ids, names, validate=False)
