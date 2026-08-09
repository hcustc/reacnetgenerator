# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bounded-memory validation for normalized timed-output HDF5 files."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

_MANIFEST_VERSION = 1
_SCHEMA_VERSION = "2"
_SUPPORTED_SCHEMA_VERSIONS = frozenset(("1", _SCHEMA_VERSION))
_DEFAULT_BLOCK_ROWS = 4096
_DEFAULT_BLOCK_BYTES = 16 * 1024**2
_UINT256_MODULUS = 1 << 256
_UINT64_MAX = np.iinfo(np.uint64).max
_STAGE_METRICS = (
    "step3_molecule_seconds",
    "step3_matrix_seconds",
    "step3_route_seconds",
    "step3_reaction_seconds",
    "timed_output_write_seconds",
)


class TimedOutputValidationError(ValueError):
    """Report an invalid or internally inconsistent timed-output artifact."""


class _MultisetFingerprint:
    """Accumulate cryptographic record digests without retaining the records."""

    def __init__(self, domain: bytes):
        self.domain = domain
        self.count = 0
        self.xor = 0
        self.total = 0
        self.square_total = 0

    def add(self, digest: bytes) -> None:
        value = int.from_bytes(digest, "big")
        self.count += 1
        self.xor ^= value
        self.total = (self.total + value) % _UINT256_MODULUS
        self.square_total = (self.square_total + value * value) % _UINT256_MODULUS

    def digest(self) -> bytes:
        hasher = hashlib.sha256(self.domain)
        hasher.update(self.count.to_bytes(16, "big"))
        for value in (self.xor, self.total, self.square_total):
            hasher.update(value.to_bytes(32, "big"))
        return hasher.digest()

    def hexdigest(self) -> str:
        return self.digest().hex()


def _fail(message: str) -> None:
    raise TimedOutputValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _decode_string(value, label: str) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    except UnicodeDecodeError as error:
        raise TimedOutputValidationError(f"{label} is not valid UTF-8") from error


def _read_dataset_slice(dataset: h5py.Dataset, start: int, stop: int):
    """Read one explicit dataset slice; kept separate for bounded-read tests."""
    return dataset[start:stop]


def _require_dataset(
    handle: h5py.File,
    path: str,
    *,
    ndim: int = 1,
    trailing_shape: tuple[int, ...] = (),
    dtype=None,
    string: bool = False,
) -> h5py.Dataset:
    item = handle.get(path)
    _require(isinstance(item, h5py.Dataset), f"Missing dataset: {path}")
    dataset = item
    _require(dataset.ndim == ndim, f"{path} must have {ndim} dimensions")
    _require(
        dataset.shape[1:] == trailing_shape,
        f"{path} has an invalid trailing shape",
    )
    if string:
        _require(
            h5py.check_string_dtype(dataset.dtype) is not None,
            f"{path} must contain UTF-8 strings",
        )
    elif dtype is not None:
        expected = np.dtype(dtype)
        _require(
            dataset.dtype.kind == expected.kind
            and dataset.dtype.itemsize == expected.itemsize,
            f"{path} has dtype {dataset.dtype}, expected {expected}",
        )
    return dataset


def _attribute(handle: h5py.File, name: str):
    _require(name in handle.attrs, f"Missing HDF5 attribute: {name}")
    return handle.attrs[name]


def _integer_attribute(handle: h5py.File, name: str) -> int:
    value = _attribute(handle, name)
    _require(
        not isinstance(value, (bool, np.bool_)),
        f"HDF5 attribute {name} must be an integer",
    )
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TimedOutputValidationError(
            f"HDF5 attribute {name} must be an integer"
        ) from error
    _require(result >= 0, f"HDF5 attribute {name} must be non-negative")
    return result


def _boolean_attribute(handle: h5py.File, name: str) -> bool:
    value = _attribute(handle, name)
    _require(
        isinstance(value, (bool, np.bool_)),
        f"HDF5 attribute {name} must be boolean",
    )
    return bool(value)


def _float_attribute(handle: h5py.File, name: str) -> float:
    value = _attribute(handle, name)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TimedOutputValidationError(
            f"HDF5 attribute {name} must be numeric"
        ) from error
    _require(
        math.isfinite(result) and result >= 0,
        f"HDF5 attribute {name} must be finite and non-negative",
    )
    return result


def _canonical_bytes(values, dtype) -> bytes:
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.ascontiguousarray(np.asarray(values, dtype=canonical_dtype))
    return array.tobytes()


def _length_prefixed(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _record_digest(domain: bytes, *fields: bytes) -> bytes:
    hasher = hashlib.sha256(domain)
    for field in fields:
        _length_prefixed(hasher, field)
    return hasher.digest()


def _combine_digests(domain: bytes, *digests: str) -> str:
    hasher = hashlib.sha256(domain)
    for digest in digests:
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def _hash_numeric_dataset(
    dataset: h5py.Dataset,
    dtype,
    block_rows: int,
    domain: bytes,
) -> str:
    hasher = hashlib.sha256(domain)
    hasher.update(len(dataset).to_bytes(8, "big"))
    for start in range(0, len(dataset), block_rows):
        stop = min(start + block_rows, len(dataset))
        hasher.update(
            _canonical_bytes(_read_dataset_slice(dataset, start, stop), dtype)
        )
    return hasher.hexdigest()


def _string_digest(value, domain: bytes, label: str) -> bytes:
    encoded = _decode_string(value, label).encode("utf-8")
    return _record_digest(domain, encoded)


def _build_string_digest_index(
    dataset: h5py.Dataset,
    block_rows: int,
    *,
    domain: bytes,
    label: str,
) -> np.ndarray:
    digests = np.empty((len(dataset), 32), dtype=np.uint8)
    for start in range(0, len(dataset), block_rows):
        stop = min(start + block_rows, len(dataset))
        values = _read_dataset_slice(dataset, start, stop)
        for offset, value in enumerate(values):
            digest = _string_digest(value, domain, label)
            digests[start + offset] = np.frombuffer(digest, dtype=np.uint8)
    _require_unique_digests(digests, f"{label} contains duplicate values")
    return digests


def _require_unique_digests(digests: np.ndarray, message: str) -> None:
    if len(digests) < 2:
        return
    packed = digests.view("V32").reshape((-1,))
    order = np.argsort(packed)
    _require(
        not np.any(packed[order[1:]] == packed[order[:-1]]),
        message,
    )


def _validate_offsets(
    dataset: h5py.Dataset,
    expected_rows: int,
    terminal: int,
    block_rows: int,
    label: str,
) -> None:
    _require(
        len(dataset) == expected_rows + 1,
        f"{label} length must equal row count + 1",
    )
    previous = None
    for start in range(0, len(dataset), block_rows):
        stop = min(start + block_rows, len(dataset))
        values = np.asarray(
            _read_dataset_slice(dataset, start, stop),
            dtype=np.uint64,
        )
        if previous is not None:
            _require(int(values[0]) >= previous, f"{label} must be nondecreasing")
        if len(values) > 1:
            _require(
                not np.any(values[1:] < values[:-1]),
                f"{label} must be nondecreasing",
            )
        if len(values):
            previous = int(values[-1])
    first = int(dataset[0])
    last = int(dataset[-1])
    _require(first == 0, f"{label} must start at zero")
    _require(last == terminal, f"{label} terminal offset is inconsistent")


class _RangeCursor:
    """Stream molecule ranges grouped by molecule ID."""

    def __init__(
        self,
        molecule_ids: h5py.Dataset,
        starts: h5py.Dataset,
        ends: h5py.Dataset,
        *,
        molecule_count: int,
        frame_count: int,
        block_rows: int,
    ):
        self.datasets = (molecule_ids, starts, ends)
        self.total_rows = len(molecule_ids)
        self.molecule_count = molecule_count
        self.frame_count = frame_count
        self.block_rows = block_rows
        self.file_position = 0
        self.block_position = 0
        self.block_ids = np.empty(0, dtype=np.uint64)
        self.block_starts = np.empty(0, dtype=np.uint64)
        self.block_ends = np.empty(0, dtype=np.uint64)
        self.previous_id = 0
        self.logical_rows = 0

    def _load(self) -> bool:
        if self.block_position < len(self.block_ids):
            return True
        if self.file_position >= self.total_rows:
            return False
        stop = min(self.file_position + self.block_rows, self.total_rows)
        self.block_ids = np.asarray(
            _read_dataset_slice(self.datasets[0], self.file_position, stop),
            dtype=np.uint64,
        )
        self.block_starts = np.asarray(
            _read_dataset_slice(self.datasets[1], self.file_position, stop),
            dtype=np.uint64,
        )
        self.block_ends = np.asarray(
            _read_dataset_slice(self.datasets[2], self.file_position, stop),
            dtype=np.uint64,
        )
        _require(
            np.all((self.block_ids >= 1) & (self.block_ids <= self.molecule_count)),
            "molecule_ranges/molecule_id is out of range",
        )
        _require(
            np.all(self.block_starts <= self.block_ends),
            "molecule_ranges contains start_frame > end_frame",
        )
        _require(
            np.all(self.block_ends < self.frame_count),
            "molecule_ranges contains an out-of-range frame",
        )
        if len(self.block_ids):
            _require(
                int(self.block_ids[0]) >= self.previous_id,
                "molecule_ranges/molecule_id must be nondecreasing",
            )
            _require(
                not np.any(self.block_ids[1:] < self.block_ids[:-1]),
                "molecule_ranges/molecule_id must be nondecreasing",
            )
            self.previous_id = int(self.block_ids[-1])
        self.file_position = stop
        self.block_position = 0
        return True

    def fingerprint_for(self, molecule_id: int) -> tuple[bytes, int]:
        accumulator = _MultisetFingerprint(b"rng:timed-output:ranges:v1\0")
        count = 0
        canonical_start = None
        canonical_end = None
        while self._load():
            current_id = int(self.block_ids[self.block_position])
            _require(
                current_id >= molecule_id,
                "molecule_ranges rows are not grouped with molecule definitions",
            )
            if current_id != molecule_id:
                break
            start = int(self.block_starts[self.block_position])
            end = int(self.block_ends[self.block_position])
            if canonical_end is not None:
                _require(
                    start > canonical_end,
                    "molecule_ranges overlap or are not ordered within a molecule",
                )
            if canonical_end is not None and start == canonical_end + 1:
                canonical_end = end
                self.block_position += 1
                continue
            if canonical_start is not None:
                accumulator.add(
                    _record_digest(
                        b"rng:timed-output:range-row:v1\0",
                        canonical_start.to_bytes(8, "little"),
                        canonical_end.to_bytes(8, "little"),
                    )
                )
                self.logical_rows += canonical_end - canonical_start + 1
                count += 1
            canonical_start = start
            canonical_end = end
            self.block_position += 1
        if canonical_start is not None:
            accumulator.add(
                _record_digest(
                    b"rng:timed-output:range-row:v1\0",
                    canonical_start.to_bytes(8, "little"),
                    canonical_end.to_bytes(8, "little"),
                )
            )
            self.logical_rows += canonical_end - canonical_start + 1
            count += 1
        return accumulator.digest(), count

    def require_exhausted(self) -> None:
        _require(not self._load(), "molecule_ranges contains unreferenced rows")


def _validate_sources(
    handle: h5py.File,
    block_rows: int,
) -> tuple[int, str]:
    paths = _require_dataset(handle, "sources/path", string=True)
    ordinals = _require_dataset(handle, "sources/ordinal", dtype=np.uint32)
    _require(len(paths) == len(ordinals), "sources datasets have unequal lengths")
    path_values = []
    path_hasher = hashlib.sha256(b"rng:timed-output:source-paths:v1\0")
    path_hasher.update(len(paths).to_bytes(8, "big"))
    for start in range(0, len(paths), block_rows):
        stop = min(start + block_rows, len(paths))
        values = _read_dataset_slice(paths, start, stop)
        for offset, value in enumerate(values):
            decoded = _decode_string(value, "sources/path")
            path_values.append(decoded)
            _length_prefixed(path_hasher, decoded.encode("utf-8"))
        ordinal_values = np.asarray(
            _read_dataset_slice(ordinals, start, stop),
            dtype=np.uint32,
        )
        expected = np.arange(start, stop, dtype=np.uint32)
        _require(
            np.array_equal(ordinal_values, expected),
            "sources/ordinal must be sequential and zero-based",
        )
    try:
        source_order = json.loads(
            _decode_string(_attribute(handle, "source_order"), "source_order")
        )
    except json.JSONDecodeError as error:
        raise TimedOutputValidationError("source_order is not valid JSON") from error
    _require(source_order == path_values, "source_order disagrees with sources/path")
    ordinal_digest = _hash_numeric_dataset(
        ordinals,
        np.uint32,
        block_rows,
        b"rng:timed-output:source-ordinals:v1\0",
    )
    fingerprint = _combine_digests(
        b"rng:timed-output:sources:v1\0",
        path_hasher.hexdigest(),
        ordinal_digest,
    )
    return len(paths), fingerprint


def _validate_frames(
    handle: h5py.File,
    *,
    frame_count: int,
    source_count: int,
    block_rows: int,
) -> str:
    source_ids = _require_dataset(handle, "frames/source_id", dtype=np.uint32)
    source_frames = _require_dataset(handle, "frames/source_frame", dtype=np.uint64)
    timesteps = _require_dataset(handle, "frames/timestep", dtype=np.int64)
    for dataset in (source_ids, source_frames, timesteps):
        _require(len(dataset) == frame_count, f"{dataset.name} length != frame_count")
    _require(
        frame_count == 0 or source_count > 0,
        "frames require at least one source",
    )
    for start in range(0, frame_count, block_rows):
        stop = min(start + block_rows, frame_count)
        values = np.asarray(
            _read_dataset_slice(source_ids, start, stop),
            dtype=np.uint32,
        )
        _require(
            np.all((values >= 1) & (values <= source_count)),
            "frames/source_id is out of range",
        )
    digests = (
        _hash_numeric_dataset(
            source_ids,
            np.uint32,
            block_rows,
            b"rng:timed-output:frame-source-id:v1\0",
        ),
        _hash_numeric_dataset(
            source_frames,
            np.uint64,
            block_rows,
            b"rng:timed-output:frame-source-frame:v1\0",
        ),
        _hash_numeric_dataset(
            timesteps,
            np.int64,
            block_rows,
            b"rng:timed-output:frame-timestep:v1\0",
        ),
    )
    return _combine_digests(b"rng:timed-output:frames:v1\0", *digests)


def _update_dataset_bytes(
    hasher,
    dataset: h5py.Dataset,
    start: int,
    stop: int,
    dtype,
    *,
    row_bytes: int,
    block_bytes: int,
) -> None:
    rows_per_read = max(1, block_bytes // row_bytes)
    for offset in range(start, stop, rows_per_read):
        block_stop = min(offset + rows_per_read, stop)
        hasher.update(
            _canonical_bytes(
                _read_dataset_slice(dataset, offset, block_stop),
                dtype,
            )
        )


def _molecule_digest_from_memory(
    species_digest: bytes,
    atoms,
    bond_atoms,
    bond_orders,
    range_digest: bytes,
    range_count: int,
) -> bytes:
    hasher = hashlib.sha256(b"rng:timed-output:molecule:v1\0")
    hasher.update(species_digest)
    hasher.update(len(atoms).to_bytes(8, "little"))
    hasher.update(_canonical_bytes(atoms, np.uint64))
    hasher.update(len(bond_orders).to_bytes(8, "little"))
    hasher.update(_canonical_bytes(bond_atoms, np.uint64))
    hasher.update(_canonical_bytes(bond_orders, np.int16))
    hasher.update(range_count.to_bytes(8, "little"))
    hasher.update(range_digest)
    return hasher.digest()


def _molecule_digest_from_datasets(
    species_digest: bytes,
    atom_ids: h5py.Dataset,
    atom_start: int,
    atom_stop: int,
    bond_atoms: h5py.Dataset,
    bond_orders: h5py.Dataset,
    bond_start: int,
    bond_stop: int,
    range_digest: bytes,
    range_count: int,
    block_bytes: int,
) -> bytes:
    hasher = hashlib.sha256(b"rng:timed-output:molecule:v1\0")
    hasher.update(species_digest)
    hasher.update((atom_stop - atom_start).to_bytes(8, "little"))
    _update_dataset_bytes(
        hasher,
        atom_ids,
        atom_start,
        atom_stop,
        np.uint64,
        row_bytes=8,
        block_bytes=block_bytes,
    )
    hasher.update((bond_stop - bond_start).to_bytes(8, "little"))
    _update_dataset_bytes(
        hasher,
        bond_atoms,
        bond_start,
        bond_stop,
        np.uint64,
        row_bytes=16,
        block_bytes=block_bytes,
    )
    _update_dataset_bytes(
        hasher,
        bond_orders,
        bond_start,
        bond_stop,
        np.int16,
        row_bytes=2,
        block_bytes=block_bytes,
    )
    hasher.update(range_count.to_bytes(8, "little"))
    hasher.update(range_digest)
    return hasher.digest()


def _validate_molecules(
    handle: h5py.File,
    *,
    molecule_count: int,
    molecule_range_count: int,
    frame_count: int,
    block_rows: int,
    block_bytes: int,
) -> tuple[str, dict[str, int], int, np.ndarray]:
    species_names = _require_dataset(handle, "species/name", string=True)
    molecule_ids = _require_dataset(handle, "molecules/molecule_id", dtype=np.uint64)
    species_ids = _require_dataset(handle, "molecules/species_id", dtype=np.uint32)
    atom_offsets = _require_dataset(handle, "molecules/atom_offsets", dtype=np.uint64)
    bond_offsets = _require_dataset(handle, "molecules/bond_offsets", dtype=np.uint64)
    atom_ids = _require_dataset(handle, "molecules/atom_ids", dtype=np.uint64)
    bond_atoms = _require_dataset(
        handle,
        "molecules/bond_atoms",
        ndim=2,
        trailing_shape=(2,),
        dtype=np.uint64,
    )
    bond_orders = _require_dataset(handle, "molecules/bond_order", dtype=np.int16)
    range_ids = _require_dataset(
        handle,
        "molecule_ranges/molecule_id",
        dtype=np.uint64,
    )
    range_starts = _require_dataset(
        handle,
        "molecule_ranges/start_frame",
        dtype=np.uint64,
    )
    range_ends = _require_dataset(
        handle,
        "molecule_ranges/end_frame",
        dtype=np.uint64,
    )
    _require(
        len(molecule_ids) == molecule_count,
        "molecules/molecule_id length != molecule_count",
    )
    _require(
        len(species_ids) == molecule_count,
        "molecules/species_id length != molecule_count",
    )
    _require(
        len(bond_atoms) == len(bond_orders),
        "molecules bond datasets have unequal lengths",
    )
    _require(
        len(range_ids) == len(range_starts) == len(range_ends),
        "molecule_ranges datasets have unequal lengths",
    )
    _require(
        len(range_ids) == molecule_range_count,
        "molecule_ranges length != molecule_range_count",
    )
    _validate_offsets(
        atom_offsets,
        molecule_count,
        len(atom_ids),
        block_rows,
        "molecules/atom_offsets",
    )
    _validate_offsets(
        bond_offsets,
        molecule_count,
        len(bond_atoms),
        block_rows,
        "molecules/bond_offsets",
    )
    species_digests = _build_string_digest_index(
        species_names,
        block_rows,
        domain=b"rng:timed-output:species:v1\0",
        label="species/name",
    )
    species_seen = np.zeros(len(species_names), dtype=np.bool_)
    ranges = _RangeCursor(
        range_ids,
        range_starts,
        range_ends,
        molecule_count=molecule_count,
        frame_count=frame_count,
        block_rows=block_rows,
    )
    accumulator = _MultisetFingerprint(b"rng:timed-output:molecules:v1\0")
    molecule_digests = np.empty((molecule_count, 32), dtype=np.uint8)
    canonical_range_count = 0

    for start in range(0, molecule_count, block_rows):
        stop = min(start + block_rows, molecule_count)
        ids = np.asarray(
            _read_dataset_slice(molecule_ids, start, stop),
            dtype=np.uint64,
        )
        expected_ids = np.arange(start + 1, stop + 1, dtype=np.uint64)
        _require(
            np.array_equal(ids, expected_ids),
            "molecules/molecule_id must be sequential and one-based",
        )
        species = np.asarray(
            _read_dataset_slice(species_ids, start, stop),
            dtype=np.uint32,
        )
        _require(
            np.all((species >= 1) & (species <= len(species_names))),
            "molecules/species_id is out of range",
        )
        atom_index = np.asarray(
            _read_dataset_slice(atom_offsets, start, stop + 1),
            dtype=np.uint64,
        )
        bond_index = np.asarray(
            _read_dataset_slice(bond_offsets, start, stop + 1),
            dtype=np.uint64,
        )
        local_start = 0
        while local_start < stop - start:
            local_stop = local_start + 1
            atom_base = int(atom_index[local_start])
            bond_base = int(bond_index[local_start])
            while local_stop < stop - start:
                atom_bytes = (int(atom_index[local_stop + 1]) - atom_base) * 8
                bond_bytes = (int(bond_index[local_stop + 1]) - bond_base) * 18
                if atom_bytes + bond_bytes > block_bytes:
                    break
                local_stop += 1
            atom_end = int(atom_index[local_stop])
            bond_end = int(bond_index[local_stop])
            oversized = (atom_end - atom_base) * 8 + (bond_end - bond_base) * 18
            if local_stop == local_start + 1 and oversized > block_bytes:
                row = local_start
                molecule_id = start + row + 1
                species_index = int(species[row]) - 1
                species_seen[species_index] = True
                range_digest, range_count = ranges.fingerprint_for(molecule_id)
                canonical_range_count += range_count
                molecule_digest = _molecule_digest_from_datasets(
                    species_digests[species_index].tobytes(),
                    atom_ids,
                    int(atom_index[row]),
                    int(atom_index[row + 1]),
                    bond_atoms,
                    bond_orders,
                    int(bond_index[row]),
                    int(bond_index[row + 1]),
                    range_digest,
                    range_count,
                    block_bytes,
                )
                accumulator.add(molecule_digest)
                molecule_digests[molecule_id - 1] = np.frombuffer(
                    molecule_digest,
                    dtype=np.uint8,
                )
                local_start = local_stop
                continue
            atoms = _read_dataset_slice(atom_ids, atom_base, atom_end)
            pairs = _read_dataset_slice(bond_atoms, bond_base, bond_end)
            orders = _read_dataset_slice(bond_orders, bond_base, bond_end)
            for row in range(local_start, local_stop):
                molecule_id = start + row + 1
                species_index = int(species[row]) - 1
                species_seen[species_index] = True
                range_digest, range_count = ranges.fingerprint_for(molecule_id)
                canonical_range_count += range_count
                atom_slice = slice(
                    int(atom_index[row]) - atom_base,
                    int(atom_index[row + 1]) - atom_base,
                )
                bond_slice = slice(
                    int(bond_index[row]) - bond_base,
                    int(bond_index[row + 1]) - bond_base,
                )
                molecule_digest = _molecule_digest_from_memory(
                    species_digests[species_index].tobytes(),
                    atoms[atom_slice],
                    pairs[bond_slice],
                    orders[bond_slice],
                    range_digest,
                    range_count,
                )
                accumulator.add(molecule_digest)
                molecule_digests[molecule_id - 1] = np.frombuffer(
                    molecule_digest,
                    dtype=np.uint8,
                )
            local_start = local_stop
    ranges.require_exhausted()
    _require(
        np.all(species_seen),
        "species/name contains definitions not referenced by molecules",
    )
    return (
        accumulator.hexdigest(),
        {
            "species_count": len(species_names),
            "atom_count": len(atom_ids),
            "bond_count": len(bond_atoms),
            "logical_molecule_row_count": ranges.logical_rows,
        },
        canonical_range_count,
        molecule_digests,
    )


def _build_reaction_type_index(
    reactants: h5py.Dataset,
    products: h5py.Dataset,
    block_rows: int,
) -> np.ndarray:
    digests = np.empty((len(reactants), 32), dtype=np.uint8)
    for start in range(0, len(reactants), block_rows):
        stop = min(start + block_rows, len(reactants))
        reactant_values = _read_dataset_slice(reactants, start, stop)
        product_values = _read_dataset_slice(products, start, stop)
        for offset, (reactant, product) in enumerate(
            zip(reactant_values, product_values)
        ):
            digest = _record_digest(
                b"rng:timed-output:reaction-type:v1\0",
                _decode_string(reactant, "reaction_types/reactant").encode("utf-8"),
                _decode_string(product, "reaction_types/product").encode("utf-8"),
            )
            digests[start + offset] = np.frombuffer(digest, dtype=np.uint8)
    _require_unique_digests(digests, "reaction_types contains duplicate pairs")
    return digests


def _validate_reaction_blocks(
    transition_indices: h5py.Dataset,
    reaction_ids: h5py.Dataset,
    block_starts: h5py.Dataset,
    block_lengths: h5py.Dataset,
    *,
    event_count: int,
    reaction_type_count: int,
    block_rows: int,
) -> None:
    covered_rows = 0
    seen_stamp = np.zeros(reaction_type_count, dtype=np.uint64)
    transition_count = len(block_starts)
    for block_offset in range(0, transition_count, block_rows):
        block_stop = min(block_offset + block_rows, transition_count)
        starts = np.asarray(
            _read_dataset_slice(block_starts, block_offset, block_stop),
            dtype=np.uint64,
        )
        lengths = np.asarray(
            _read_dataset_slice(block_lengths, block_offset, block_stop),
            dtype=np.uint64,
        )
        for local_index, (start_value, length_value) in enumerate(zip(starts, lengths)):
            transition = block_offset + local_index
            start = int(start_value)
            length = int(length_value)
            _require(
                start <= event_count, "reaction_events/block_start is out of range"
            )
            _require(
                length <= event_count - start,
                "reaction_events block exceeds the event table",
            )
            covered_rows += length
            stamp = transition + 1
            for row_start in range(start, start + length, block_rows):
                row_stop = min(row_start + block_rows, start + length)
                transitions = np.asarray(
                    _read_dataset_slice(
                        transition_indices,
                        row_start,
                        row_stop,
                    ),
                    dtype=np.uint64,
                )
                _require(
                    np.all(transitions == transition),
                    "reaction_events block disagrees with transition_index",
                )
                ids = np.asarray(
                    _read_dataset_slice(reaction_ids, row_start, row_stop),
                    dtype=np.uint32,
                ).astype(np.int64, copy=False)
                indices = ids - 1
                _require(
                    len(np.unique(indices)) == len(indices),
                    "reaction_events contains a duplicate reaction in one transition",
                )
                _require(
                    not np.any(seen_stamp[indices] == stamp),
                    "reaction_events contains a duplicate reaction in one transition",
                )
                seen_stamp[indices] = stamp
    _require(
        covered_rows == event_count,
        "reaction_events blocks do not cover the event table exactly",
    )


def _validate_reactions(
    handle: h5py.File,
    *,
    frame_count: int,
    reaction_type_count: int,
    reaction_event_row_count: int,
    block_rows: int,
) -> tuple[str, int, np.ndarray]:
    reactants = _require_dataset(handle, "reaction_types/reactant", string=True)
    products = _require_dataset(handle, "reaction_types/product", string=True)
    totals = _require_dataset(
        handle,
        "reaction_types/total_count",
        dtype=np.uint64,
    )
    transition_indices = _require_dataset(
        handle,
        "reaction_events/transition_index",
        dtype=np.uint64,
    )
    reaction_ids = _require_dataset(
        handle,
        "reaction_events/reaction_id",
        dtype=np.uint32,
    )
    counts = _require_dataset(handle, "reaction_events/count", dtype=np.uint64)
    block_starts = _require_dataset(
        handle,
        "reaction_events/block_start",
        dtype=np.uint64,
    )
    block_lengths = _require_dataset(
        handle,
        "reaction_events/block_length",
        dtype=np.uint64,
    )
    _require(
        len(reactants) == len(products) == len(totals) == reaction_type_count,
        "reaction_types lengths disagree with reaction_type_count",
    )
    _require(
        len(transition_indices)
        == len(reaction_ids)
        == len(counts)
        == reaction_event_row_count,
        "reaction_events lengths disagree with reaction_event_row_count",
    )
    transition_count = max(0, frame_count - 1)
    _require(
        len(block_starts) == len(block_lengths) == transition_count,
        "reaction_events block index length != frame_count - 1",
    )
    type_digests = _build_reaction_type_index(reactants, products, block_rows)
    stored_totals = np.empty(reaction_type_count, dtype=np.uint64)
    for start in range(0, reaction_type_count, block_rows):
        stop = min(start + block_rows, reaction_type_count)
        stored_totals[start:stop] = _read_dataset_slice(totals, start, stop)
    _require(
        np.all(stored_totals > 0),
        "reaction_types/total_count must be positive",
    )
    observed_totals = np.zeros(reaction_type_count, dtype=np.uint64)
    accumulator = _MultisetFingerprint(b"rng:timed-output:reaction-events:v1\0")
    logical_event_count = 0
    for start in range(0, reaction_event_row_count, block_rows):
        stop = min(start + block_rows, reaction_event_row_count)
        transitions = np.asarray(
            _read_dataset_slice(transition_indices, start, stop),
            dtype=np.uint64,
        )
        ids = np.asarray(
            _read_dataset_slice(reaction_ids, start, stop),
            dtype=np.uint32,
        )
        values = np.asarray(
            _read_dataset_slice(counts, start, stop),
            dtype=np.uint64,
        )
        _require(
            np.all(transitions < transition_count),
            "reaction_events/transition_index is out of range",
        )
        _require(
            np.all((ids >= 1) & (ids <= reaction_type_count)),
            "reaction_events/reaction_id is out of range",
        )
        _require(np.all(values > 0), "reaction_events/count must be positive")
        for transition, reaction_id, count in zip(transitions, ids, values):
            type_index = int(reaction_id) - 1
            old_total = int(observed_totals[type_index])
            count_value = int(count)
            _require(
                old_total <= _UINT64_MAX - count_value,
                "reaction total exceeds uint64 capacity",
            )
            observed_totals[type_index] = old_total + count_value
            logical_event_count += count_value
            accumulator.add(
                _record_digest(
                    b"rng:timed-output:reaction-event:v1\0",
                    int(transition).to_bytes(8, "little"),
                    type_digests[type_index].tobytes(),
                    count_value.to_bytes(8, "little"),
                )
            )
    _require(
        np.array_equal(observed_totals, stored_totals),
        "reaction_types/total_count disagrees with reaction_events",
    )
    _validate_reaction_blocks(
        transition_indices,
        reaction_ids,
        block_starts,
        block_lengths,
        event_count=reaction_event_row_count,
        reaction_type_count=reaction_type_count,
        block_rows=block_rows,
    )
    return accumulator.hexdigest(), logical_event_count, type_digests


def _participant_bond_map(
    molecule_ids,
    bond_offsets: h5py.Dataset,
    bond_atoms: h5py.Dataset,
    bond_orders: h5py.Dataset,
    block_rows: int,
) -> dict[tuple[int, int], int]:
    """Rebuild one side's bond map from referenced molecule definitions."""
    result: dict[tuple[int, int], int] = {}
    for molecule_id in molecule_ids:
        row = int(molecule_id) - 1
        start = int(bond_offsets[row])
        stop = int(bond_offsets[row + 1])
        for block_start in range(start, stop, block_rows):
            block_stop = min(block_start + block_rows, stop)
            pairs = np.asarray(
                _read_dataset_slice(bond_atoms, block_start, block_stop),
                dtype=np.uint64,
            )
            orders = np.asarray(
                _read_dataset_slice(bond_orders, block_start, block_stop),
                dtype=np.int16,
            )
            for pair, order in zip(pairs, orders):
                atom1, atom2 = sorted((int(pair[0]), int(pair[1])))
                key = (atom1, atom2)
                value = int(order)
                previous = result.get(key)
                _require(
                    previous is None or previous == value,
                    "transition evidence participants contain conflicting bonds",
                )
                result[key] = value
    return result


def _require_disjoint_participant_atoms(
    molecule_ids,
    atom_offsets: h5py.Dataset,
    atom_ids: h5py.Dataset,
    block_rows: int,
) -> set[int]:
    """Return one side's atoms after rejecting overlapping molecules."""
    seen: set[int] = set()
    for molecule_id in molecule_ids:
        row = int(molecule_id) - 1
        start = int(atom_offsets[row])
        stop = int(atom_offsets[row + 1])
        for block_start in range(start, stop, block_rows):
            block_stop = min(block_start + block_rows, stop)
            values = np.asarray(
                _read_dataset_slice(atom_ids, block_start, block_stop),
                dtype=np.uint64,
            )
            for atom_id in values:
                atom_id = int(atom_id)
                _require(
                    atom_id not in seen,
                    "transition evidence participants contain overlapping atoms",
                )
                seen.add(atom_id)
    return seen


def _participant_reaction_pair(
    reactants,
    products,
    molecule_species_ids: h5py.Dataset,
    species_names: h5py.Dataset,
) -> tuple[str, str] | None:
    """Recompute the filtered reaction label from exact participants."""
    sides = []
    for molecule_ids in (reactants, products):
        names = Counter()
        for molecule_id in molecule_ids:
            species_id = int(molecule_species_ids[int(molecule_id) - 1])
            name = _decode_string(
                species_names[species_id - 1],
                "species/name",
            )
            names[name] += 1
        sides.append(names)
    net_reactants = sides[0] - sides[1]
    net_products = sides[1] - sides[0]
    if not net_reactants or not net_products:
        return None
    return (
        "+".join(sorted(net_reactants.elements())),
        "+".join(sorted(net_products.elements())),
    )


def _validate_transition_evidence(
    handle: h5py.File,
    *,
    frame_count: int,
    molecule_count: int,
    reaction_type_count: int,
    reaction_event_row_count: int,
    logical_reaction_event_count: int,
    evidence_event_count: int,
    participant_count: int,
    bond_change_count: int,
    molecule_digests: np.ndarray,
    reaction_type_digests: np.ndarray,
    evidence_enabled: bool,
    block_rows: int,
) -> str:
    """Validate instance evidence and its exact relationship to compact events."""
    transitions = _require_dataset(
        handle,
        "transition_evidence/transition_index",
        dtype=np.uint64,
    )
    reaction_ids = _require_dataset(
        handle,
        "transition_evidence/reaction_id",
        dtype=np.uint32,
    )
    participant_offsets = _require_dataset(
        handle,
        "transition_evidence/participant_offsets",
        dtype=np.uint64,
    )
    participant_ids = _require_dataset(
        handle,
        "transition_evidence/participant_molecule_id",
        dtype=np.uint64,
    )
    participant_sides = _require_dataset(
        handle,
        "transition_evidence/participant_side",
        dtype=np.uint8,
    )
    bond_offsets = _require_dataset(
        handle,
        "transition_evidence/bond_change_offsets",
        dtype=np.uint64,
    )
    bond_atoms = _require_dataset(
        handle,
        "transition_evidence/bond_atoms",
        ndim=2,
        trailing_shape=(2,),
        dtype=np.uint64,
    )
    before_orders = _require_dataset(
        handle,
        "transition_evidence/before_order",
        dtype=np.int16,
    )
    after_orders = _require_dataset(
        handle,
        "transition_evidence/after_order",
        dtype=np.int16,
    )
    block_starts = _require_dataset(
        handle,
        "transition_evidence/block_start",
        dtype=np.uint64,
    )
    block_lengths = _require_dataset(
        handle,
        "transition_evidence/block_length",
        dtype=np.uint64,
    )
    _require(
        len(transitions) == len(reaction_ids) == evidence_event_count,
        "transition_evidence event lengths disagree with the event count",
    )
    _require(
        len(participant_ids) == len(participant_sides) == participant_count,
        "transition_evidence participant lengths disagree with the participant count",
    )
    _require(
        len(bond_atoms) == len(before_orders) == len(after_orders) == bond_change_count,
        "transition_evidence bond-change lengths disagree with the bond count",
    )
    _validate_offsets(
        participant_offsets,
        evidence_event_count,
        participant_count,
        block_rows,
        "transition_evidence/participant_offsets",
    )
    _validate_offsets(
        bond_offsets,
        evidence_event_count,
        bond_change_count,
        block_rows,
        "transition_evidence/bond_change_offsets",
    )
    transition_count = max(0, frame_count - 1)
    _require(
        len(block_starts) == len(block_lengths) == transition_count,
        "transition_evidence block index length != frame_count - 1",
    )
    if not evidence_enabled:
        _require(
            evidence_event_count == participant_count == bond_change_count == 0,
            "disabled transition evidence tables must be empty",
        )
        for start in range(0, transition_count, block_rows):
            stop = min(start + block_rows, transition_count)
            starts = np.asarray(
                _read_dataset_slice(block_starts, start, stop),
                dtype=np.uint64,
            )
            lengths = np.asarray(
                _read_dataset_slice(block_lengths, start, stop),
                dtype=np.uint64,
            )
            _require(
                not np.any(starts) and not np.any(lengths),
                "disabled transition evidence blocks must be empty",
            )
        return _MultisetFingerprint(
            b"rng:timed-output:transition-evidence:v1\0"
        ).hexdigest()
    _require(
        evidence_event_count == logical_reaction_event_count,
        "transition evidence count disagrees with logical reaction events",
    )

    aggregate_events = handle["reaction_events"]
    aggregate_reaction_ids = aggregate_events["reaction_id"]
    aggregate_counts = aggregate_events["count"]
    aggregate_block_starts = aggregate_events["block_start"]
    aggregate_block_lengths = aggregate_events["block_length"]
    molecule_bond_offsets = handle["molecules/bond_offsets"]
    molecule_bond_atoms = handle["molecules/bond_atoms"]
    molecule_bond_orders = handle["molecules/bond_order"]
    molecule_atom_offsets = handle["molecules/atom_offsets"]
    molecule_atom_ids = handle["molecules/atom_ids"]
    molecule_species_ids = handle["molecules/species_id"]
    species_names = handle["species/name"]
    reaction_reactants = handle["reaction_types/reactant"]
    reaction_products = handle["reaction_types/product"]
    accumulator = _MultisetFingerprint(b"rng:timed-output:transition-evidence:v1\0")
    covered_events = 0
    for transition in range(transition_count):
        aggregate_start = int(aggregate_block_starts[transition])
        aggregate_stop = aggregate_start + int(aggregate_block_lengths[transition])
        remaining: dict[int, int] = {}
        for start in range(aggregate_start, aggregate_stop, block_rows):
            stop = min(start + block_rows, aggregate_stop)
            ids = np.asarray(
                _read_dataset_slice(aggregate_reaction_ids, start, stop),
                dtype=np.uint32,
            )
            counts = np.asarray(
                _read_dataset_slice(aggregate_counts, start, stop),
                dtype=np.uint64,
            )
            for reaction_id, count in zip(ids, counts):
                remaining[int(reaction_id)] = int(count)

        event_start = int(block_starts[transition])
        event_stop = event_start + int(block_lengths[transition])
        _require(
            event_start <= evidence_event_count and event_stop <= evidence_event_count,
            "transition_evidence block is out of range",
        )
        covered_events += event_stop - event_start
        for event_row in range(event_start, event_stop):
            stored_transition = int(transitions[event_row])
            reaction_id = int(reaction_ids[event_row])
            _require(
                stored_transition == transition,
                "transition_evidence block disagrees with transition_index",
            )
            _require(
                1 <= reaction_id <= reaction_type_count,
                "transition_evidence/reaction_id is out of range",
            )
            _require(
                remaining.get(reaction_id, 0) > 0,
                "transition evidence reactions disagree with reaction_events",
            )
            remaining[reaction_id] -= 1

            participant_start = int(participant_offsets[event_row])
            participant_stop = int(participant_offsets[event_row + 1])
            reactants: list[int] = []
            products: list[int] = []
            previous = (-1, 0)
            participant_fingerprints = (
                _MultisetFingerprint(b"rng:timed-output:transition-reactants:v1\0"),
                _MultisetFingerprint(b"rng:timed-output:transition-products:v1\0"),
            )
            event_hasher = hashlib.sha256(
                b"rng:timed-output:transition-evidence-event:v1\0"
            )
            event_hasher.update(transition.to_bytes(8, "little"))
            event_hasher.update(reaction_type_digests[reaction_id - 1].tobytes())
            for start in range(participant_start, participant_stop, block_rows):
                stop = min(start + block_rows, participant_stop)
                ids = np.asarray(
                    _read_dataset_slice(participant_ids, start, stop),
                    dtype=np.uint64,
                )
                sides = np.asarray(
                    _read_dataset_slice(participant_sides, start, stop),
                    dtype=np.uint8,
                )
                _require(
                    np.all((ids >= 1) & (ids <= molecule_count)),
                    "transition evidence participant molecule_id is out of range",
                )
                _require(
                    np.all(sides <= 1),
                    "transition evidence participant_side is invalid",
                )
                for molecule_id, side in zip(ids, sides):
                    value = (int(side), int(molecule_id))
                    _require(
                        value > previous,
                        "transition evidence participants must be unique and ordered",
                    )
                    previous = value
                    (reactants if value[0] == 0 else products).append(value[1])
                    participant_fingerprints[value[0]].add(
                        molecule_digests[value[1] - 1].tobytes()
                    )
            _require(
                bool(reactants) and bool(products),
                "transition evidence must contain both reaction sides",
            )
            expected_pair = (
                _decode_string(
                    reaction_reactants[reaction_id - 1],
                    "reaction_types/reactant",
                ),
                _decode_string(
                    reaction_products[reaction_id - 1],
                    "reaction_types/product",
                ),
            )
            _require(
                _participant_reaction_pair(
                    reactants,
                    products,
                    molecule_species_ids,
                    species_names,
                )
                == expected_pair,
                "transition evidence participant species disagree with reaction type",
            )
            for fingerprint in participant_fingerprints:
                event_hasher.update(fingerprint.digest())

            reactant_atoms = _require_disjoint_participant_atoms(
                reactants,
                molecule_atom_offsets,
                molecule_atom_ids,
                block_rows,
            )
            product_atoms = _require_disjoint_participant_atoms(
                products,
                molecule_atom_offsets,
                molecule_atom_ids,
                block_rows,
            )
            _require(
                reactant_atoms == product_atoms,
                "transition evidence participants do not conserve atoms",
            )

            before = _participant_bond_map(
                reactants,
                molecule_bond_offsets,
                molecule_bond_atoms,
                molecule_bond_orders,
                block_rows,
            )
            after = _participant_bond_map(
                products,
                molecule_bond_offsets,
                molecule_bond_atoms,
                molecule_bond_orders,
                block_rows,
            )
            expected_changes = [
                (pair[0], pair[1], before.get(pair, 0), after.get(pair, 0))
                for pair in sorted(before.keys() | after.keys())
                if before.get(pair, 0) != after.get(pair, 0)
            ]
            change_start = int(bond_offsets[event_row])
            change_stop = int(bond_offsets[event_row + 1])
            _require(
                change_stop - change_start == len(expected_changes),
                "transition evidence bond changes disagree with participants",
            )
            expected_offset = 0
            for start in range(change_start, change_stop, block_rows):
                stop = min(start + block_rows, change_stop)
                pairs = np.asarray(
                    _read_dataset_slice(bond_atoms, start, stop),
                    dtype=np.uint64,
                )
                before_values = np.asarray(
                    _read_dataset_slice(before_orders, start, stop),
                    dtype=np.int16,
                )
                after_values = np.asarray(
                    _read_dataset_slice(after_orders, start, stop),
                    dtype=np.int16,
                )
                actual = [
                    (int(pair[0]), int(pair[1]), int(before_order), int(after_order))
                    for pair, before_order, after_order in zip(
                        pairs,
                        before_values,
                        after_values,
                    )
                ]
                expected = expected_changes[
                    expected_offset : expected_offset + len(actual)
                ]
                _require(
                    actual == expected,
                    "transition evidence bond changes disagree with participants",
                )
                for atom1, atom2, before_order, after_order in actual:
                    event_hasher.update(atom1.to_bytes(8, "little"))
                    event_hasher.update(atom2.to_bytes(8, "little"))
                    event_hasher.update(before_order.to_bytes(2, "little"))
                    event_hasher.update(after_order.to_bytes(2, "little"))
                expected_offset += len(actual)
            accumulator.add(event_hasher.digest())
        _require(
            all(count == 0 for count in remaining.values()),
            "transition evidence reactions disagree with reaction_events",
        )
    _require(
        covered_events == evidence_event_count,
        "transition_evidence blocks do not cover the event table exactly",
    )
    _require(
        reaction_event_row_count == 0 or evidence_event_count > 0,
        "reaction events are present without transition evidence",
    )
    return accumulator.hexdigest()


def _comparison_projection(manifest: Mapping) -> dict:
    return {
        "manifest_version": manifest.get("manifest_version"),
        "schema_version": manifest.get("schema_version"),
        "flags": manifest.get("flags"),
        "counts": manifest.get("counts"),
        "fingerprints": manifest.get("fingerprints"),
        "semantic_fingerprint": manifest.get("semantic_fingerprint"),
    }


def build_timed_output_manifest(
    filename: str | Path,
    *,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    block_bytes: int = _DEFAULT_BLOCK_BYTES,
) -> dict:
    """Validate a completed timed-output file and return semantic fingerprints.

    The validator reads large frame, molecule, range, and event tables in bounded
    slices. It never expands molecule ranges into per-frame rows or reaction counts
    into repeated events. Molecule and reaction fingerprints are multisets, so
    nondeterministic internal ID assignment does not affect comparisons.
    """
    block_rows = operator.index(block_rows)
    block_bytes = operator.index(block_bytes)
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    path = Path(filename)
    with h5py.File(path, "r") as handle:
        status = _decode_string(_attribute(handle, "status"), "status")
        _require(status == "complete", "Timed-output HDF5 file is not complete")
        schema_version = _decode_string(
            _attribute(handle, "schema_version"),
            "schema_version",
        )
        _require(
            schema_version in _SUPPORTED_SCHEMA_VERSIONS,
            f"Unsupported timed-output schema version: {schema_version}",
        )
        has_transition_evidence = schema_version == _SCHEMA_VERSION
        frame_count = _integer_attribute(handle, "frame_count")
        stepinterval = _integer_attribute(handle, "stepinterval")
        _require(stepinterval > 0, "HDF5 attribute stepinterval must be positive")
        molecule_count = _integer_attribute(handle, "molecule_count")
        molecule_range_count = _integer_attribute(handle, "molecule_range_count")
        molecule_write_batches = _integer_attribute(
            handle,
            "molecule_write_batches",
        )
        reaction_type_count = _integer_attribute(handle, "reaction_type_count")
        reaction_event_row_count = _integer_attribute(
            handle,
            "reaction_event_row_count",
        )
        reaction_write_batches = _integer_attribute(
            handle,
            "reaction_write_batches",
        )
        molecule_enabled = _boolean_attribute(handle, "molecule_enabled")
        reaction_enabled = _boolean_attribute(handle, "reaction_enabled")
        if has_transition_evidence:
            transition_evidence_event_count = _integer_attribute(
                handle,
                "transition_evidence_event_count",
            )
            transition_participant_count = _integer_attribute(
                handle,
                "transition_participant_count",
            )
            transition_bond_change_count = _integer_attribute(
                handle,
                "transition_bond_change_count",
            )
            transition_evidence_write_batches = _integer_attribute(
                handle,
                "transition_evidence_write_batches",
            )
            maximum_evidence_batch_events = _integer_attribute(
                handle,
                "maximum_transition_evidence_batch_event_count",
            )
            maximum_evidence_batch_participants = _integer_attribute(
                handle,
                "maximum_transition_evidence_batch_participant_count",
            )
            maximum_evidence_batch_bond_changes = _integer_attribute(
                handle,
                "maximum_transition_evidence_batch_bond_change_count",
            )
            maximum_evidence_batch_bytes = _integer_attribute(
                handle,
                "maximum_transition_evidence_batch_bytes",
            )
            evidence_batch_row_limit = _integer_attribute(
                handle,
                "transition_evidence_batch_row_limit",
            )
            _require(
                evidence_batch_row_limit > 0,
                "transition_evidence_batch_row_limit must be positive",
            )
            _require(
                transition_evidence_event_count == 0
                or transition_evidence_write_batches > 0,
                "transition evidence is nonempty but write_batches is zero",
            )
            _require(
                maximum_evidence_batch_events <= transition_evidence_event_count
                and maximum_evidence_batch_participants <= transition_participant_count
                and maximum_evidence_batch_bond_changes <= transition_bond_change_count,
                "transition evidence batch high-water marks exceed table counts",
            )
            _require(
                maximum_evidence_batch_events <= evidence_batch_row_limit,
                "transition evidence event batch exceeds its row limit",
            )
            _require(
                transition_evidence_write_batches > 0
                or maximum_evidence_batch_events
                == maximum_evidence_batch_participants
                == maximum_evidence_batch_bond_changes
                == maximum_evidence_batch_bytes
                == 0,
                "empty transition evidence has nonzero batch high-water marks",
            )
            molecule_definitions_enabled = _boolean_attribute(
                handle,
                "molecule_definitions_enabled",
            )
            transition_evidence_enabled = _boolean_attribute(
                handle,
                "transition_evidence_enabled",
            )
            _require(
                molecule_definitions_enabled
                == (molecule_enabled or transition_evidence_enabled),
                "molecule_definitions_enabled disagrees with output flags",
            )
            _require(
                not transition_evidence_enabled or reaction_enabled,
                "transition evidence requires reaction output",
            )
            _require(
                molecule_enabled or molecule_range_count == 0,
                "molecule timeline is disabled but molecule ranges are nonempty",
            )
        else:
            transition_evidence_event_count = 0
            transition_participant_count = 0
            transition_bond_change_count = 0
            transition_evidence_write_batches = 0
            molecule_definitions_enabled = molecule_enabled
            transition_evidence_enabled = False
            _require(
                molecule_enabled or molecule_count == molecule_range_count == 0,
                "molecule output is disabled but molecule tables are nonempty",
            )
        _require(
            reaction_enabled or reaction_type_count == reaction_event_row_count == 0,
            "reaction output is disabled but reaction tables are nonempty",
        )
        _require(
            molecule_count == 0 or molecule_write_batches > 0,
            "molecule_count is nonzero but molecule_write_batches is zero",
        )
        _require(
            reaction_event_row_count == 0 or reaction_write_batches > 0,
            "reaction_event_row_count is nonzero but reaction_write_batches is zero",
        )
        started_at = _float_attribute(handle, "started_at")
        completed_at = _float_attribute(handle, "completed_at")
        _require(completed_at >= started_at, "completed_at precedes started_at")
        performance = {name: _float_attribute(handle, name) for name in _STAGE_METRICS}
        source_count, sources_fingerprint = _validate_sources(handle, block_rows)
        frames_fingerprint = _validate_frames(
            handle,
            frame_count=frame_count,
            source_count=source_count,
            block_rows=block_rows,
        )
        (
            molecules_fingerprint,
            molecule_counts,
            canonical_molecule_range_count,
            molecule_digests,
        ) = _validate_molecules(
            handle,
            molecule_count=molecule_count,
            molecule_range_count=molecule_range_count,
            frame_count=frame_count,
            block_rows=block_rows,
            block_bytes=block_bytes,
        )
        (
            reactions_fingerprint,
            logical_reaction_event_count,
            reaction_type_digests,
        ) = _validate_reactions(
            handle,
            frame_count=frame_count,
            reaction_type_count=reaction_type_count,
            reaction_event_row_count=reaction_event_row_count,
            block_rows=block_rows,
        )
        if has_transition_evidence:
            transition_evidence_fingerprint = _validate_transition_evidence(
                handle,
                frame_count=frame_count,
                molecule_count=molecule_count,
                reaction_type_count=reaction_type_count,
                reaction_event_row_count=reaction_event_row_count,
                logical_reaction_event_count=logical_reaction_event_count,
                evidence_event_count=transition_evidence_event_count,
                participant_count=transition_participant_count,
                bond_change_count=transition_bond_change_count,
                molecule_digests=molecule_digests,
                reaction_type_digests=reaction_type_digests,
                evidence_enabled=transition_evidence_enabled,
                block_rows=block_rows,
            )
        counts = {
            "source_count": source_count,
            "frame_count": frame_count,
            "molecule_count": molecule_count,
            "molecule_range_count": molecule_range_count,
            **molecule_counts,
            "reaction_type_count": reaction_type_count,
            "reaction_event_row_count": reaction_event_row_count,
            "logical_reaction_event_count": logical_reaction_event_count,
        }
        if has_transition_evidence:
            counts.update(
                {
                    "transition_evidence_event_count": (
                        transition_evidence_event_count
                    ),
                    "transition_participant_count": transition_participant_count,
                    "transition_bond_change_count": transition_bond_change_count,
                }
            )
        fingerprints = {
            "sources": sources_fingerprint,
            "frames": frames_fingerprint,
            "molecules": molecules_fingerprint,
            "reactions": reactions_fingerprint,
        }
        if has_transition_evidence:
            fingerprints["transition_evidence"] = transition_evidence_fingerprint
        semantic_counts = dict(counts)
        semantic_counts["molecule_range_count"] = canonical_molecule_range_count
        semantic_payload = {
            "schema_version": schema_version,
            "flags": {
                "molecule_enabled": molecule_enabled,
                "reaction_enabled": reaction_enabled,
            },
            "counts": semantic_counts,
            "fingerprints": {
                key: fingerprints[key]
                for key in (
                    ("frames", "molecules", "reactions", "transition_evidence")
                    if has_transition_evidence
                    else ("frames", "molecules", "reactions")
                )
            },
        }
        if has_transition_evidence:
            semantic_payload["flags"].update(
                {
                    "molecule_definitions_enabled": molecule_definitions_enabled,
                    "transition_evidence_enabled": transition_evidence_enabled,
                }
            )
        semantic_fingerprint = hashlib.sha256(
            json.dumps(
                semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "manifest_version": _MANIFEST_VERSION,
            "schema_version": schema_version,
            "status": status,
            "file_size_bytes": path.stat().st_size,
            "run": {
                "job_id": _decode_string(_attribute(handle, "job_id"), "job_id"),
                "reacnetgenerator_version": _decode_string(
                    _attribute(handle, "reacnetgenerator_version"),
                    "reacnetgenerator_version",
                ),
                "started_at": started_at,
                "completed_at": completed_at,
                "stepinterval": stepinterval,
            },
            "flags": semantic_payload["flags"],
            "counts": counts,
            "performance_seconds": performance,
            "write_batches": {
                "molecules": molecule_write_batches,
                "reactions": reaction_write_batches,
                **(
                    {
                        "transition_evidence": transition_evidence_write_batches,
                    }
                    if has_transition_evidence
                    else {}
                ),
            },
            "fingerprints": fingerprints,
            "semantic_fingerprint": semantic_fingerprint,
        }


def compare_timed_output_manifests(
    candidate: Mapping,
    baseline: Mapping,
    *,
    include_provenance: bool = False,
) -> list[str]:
    """Return semantic differences between a candidate and baseline manifest.

    Source paths are provenance rather than result semantics and are ignored by
    default. Set ``include_provenance`` to require identical stored source paths.
    """
    if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("candidate and baseline manifests must be JSON objects")
    candidate_projection = _comparison_projection(candidate)
    baseline_projection = _comparison_projection(baseline)
    mismatches = []
    for key in ("manifest_version", "schema_version", "flags"):
        if candidate_projection[key] != baseline_projection[key]:
            mismatches.append(
                f"{key}: candidate={candidate_projection[key]!r}, "
                f"baseline={baseline_projection[key]!r}"
            )
    for section in ("counts", "fingerprints"):
        candidate_values = candidate_projection.get(section) or {}
        baseline_values = baseline_projection.get(section) or {}
        keys = set(candidate_values) | set(baseline_values)
        if section == "fingerprints" and not include_provenance:
            keys.discard("sources")
        if section == "counts":
            # Adjacent ranges are a storage representation detail. Their
            # canonical coverage is already included in both fingerprints.
            keys.discard("molecule_range_count")
        for key in sorted(keys):
            if candidate_values.get(key) != baseline_values.get(key):
                mismatches.append(
                    f"{section}.{key}: candidate={candidate_values.get(key)!r}, "
                    f"baseline={baseline_values.get(key)!r}"
                )
    if (
        candidate_projection["semantic_fingerprint"]
        != baseline_projection["semantic_fingerprint"]
    ):
        mismatches.append(
            "semantic_fingerprint: "
            f"candidate={candidate_projection['semantic_fingerprint']!r}, "
            f"baseline={baseline_projection['semantic_fingerprint']!r}"
        )
    return mismatches
