# SPDX-License-Identifier: LGPL-3.0-or-later
"""Useful methods to futhur process ReacNetGenerator results."""

import heapq
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import ase.geometry
import ase.units
import h5py
import numpy as np

from ._timedoutputvalidate import (
    TimedOutputValidationError as TimedOutputValidationError,
)
from ._timedoutputvalidate import (
    build_timed_output_manifest as build_timed_output_manifest,
)
from ._timedoutputvalidate import (
    compare_timed_output_manifests as compare_timed_output_manifests,
)


@dataclass(frozen=True)
class TransitionParticipant:
    """One concrete molecule instance participating on one reaction side."""

    side: str
    molecule_id: int
    species: str
    atom_ids: tuple[int, ...]
    bonds: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class BondChange:
    """An inferred connectivity change for one canonical atom pair."""

    atom1: int
    atom2: int
    before_order: int
    after_order: int

    @property
    def kind(self) -> str:
        """Classify the change without discarding its before/after bond orders."""
        if self.before_order == 0:
            return "formed"
        if self.after_order == 0:
            return "broken"
        return "order_changed"


@dataclass(frozen=True)
class TransitionEvidence:
    """Auditable evidence for one inferred connected reaction instance."""

    transition_index: int
    reactant: str
    product: str
    participants: tuple[TransitionParticipant, ...]
    bond_changes: tuple[BondChange, ...]


def _decode_hdf5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _open_timed_output(filename: str | Path) -> h5py.File:
    handle = h5py.File(filename, "r")
    if _decode_hdf5_string(handle.attrs.get("status", "")) != "complete":
        handle.close()
        raise ValueError("Timed-output HDF5 file is not complete")
    return handle


def read_timed_output_metadata(filename: str | Path) -> dict[str, str]:
    """Read metadata from a completed timed-output HDF5 file."""
    with _open_timed_output(filename) as handle:
        return {
            str(key): _decode_hdf5_string(value) for key, value in handle.attrs.items()
        }


def iter_molecule_timeline(
    filename: str | Path,
) -> Iterator[tuple[int, str, str, str]]:
    """Lazily expand molecule ranges into the historical logical row format."""
    with _open_timed_output(filename) as handle:
        molecules = handle["molecules"]
        ranges = handle["molecule_ranges"]
        species_names = handle["species/name"][:]
        molecule_rows = {
            int(molecule_id): row
            for row, molecule_id in enumerate(molecules["molecule_id"][:])
        }
        atom_offsets = molecules["atom_offsets"][:]
        bond_offsets = molecules["bond_offsets"][:]
        heap = []
        for molecule_id, start_frame, end_frame in zip(
            ranges["molecule_id"][:],
            ranges["start_frame"][:],
            ranges["end_frame"][:],
        ):
            molecule_id = int(molecule_id)
            row = molecule_rows[molecule_id]
            atom_start, atom_end = map(int, atom_offsets[row : row + 2])
            bond_start, bond_end = map(int, bond_offsets[row : row + 2])
            atoms = molecules["atom_ids"][atom_start:atom_end]
            bond_atoms = molecules["bond_atoms"][bond_start:bond_end]
            bond_order = molecules["bond_order"][bond_start:bond_end]
            atom_ids = ";".join(str(int(atom)) for atom in atoms)
            bond_ids = ";".join(
                f"{int(pair[0])}-{int(pair[1])}-{int(order)}"
                for pair, order in zip(bond_atoms, bond_order)
            )
            species_id = int(molecules["species_id"][row])
            species = _decode_hdf5_string(species_names[species_id - 1])
            heapq.heappush(
                heap,
                (
                    int(start_frame),
                    molecule_id,
                    int(end_frame),
                    species,
                    atom_ids,
                    bond_ids,
                ),
            )
        timesteps = handle["frames/timestep"]
        while heap:
            frame, molecule_id, end_frame, species, atom_ids, bond_ids = heapq.heappop(
                heap
            )
            yield int(timesteps[frame]), species, atom_ids, bond_ids
            if frame < end_frame:
                heapq.heappush(
                    heap,
                    (
                        frame + 1,
                        molecule_id,
                        end_frame,
                        species,
                        atom_ids,
                        bond_ids,
                    ),
                )


def iter_reaction_events(
    filename: str | Path,
) -> Iterator[tuple[int, str, str]]:
    """Lazily expand aggregated reaction events into logical event rows."""
    with _open_timed_output(filename) as handle:
        reactants = handle["reaction_types/reactant"][:]
        products = handle["reaction_types/product"][:]
        events = handle["reaction_events"]
        for transition_index, (block_start, block_length) in enumerate(
            zip(events["block_start"], events["block_length"])
        ):
            block_start = int(block_start)
            block_stop = block_start + int(block_length)
            for reaction_id, count in zip(
                events["reaction_id"][block_start:block_stop],
                events["count"][block_start:block_stop],
            ):
                reactant = _decode_hdf5_string(reactants[int(reaction_id) - 1])
                product = _decode_hdf5_string(products[int(reaction_id) - 1])
                for _ in range(int(count)):
                    yield transition_index, reactant, product


def _read_transition_participant(
    handle: h5py.File,
    molecule_id: int,
    side: str,
) -> TransitionParticipant:
    molecules = handle["molecules"]
    row = int(molecule_id) - 1
    if row < 0 or row >= len(molecules["molecule_id"]):
        raise ValueError("Transition evidence references an invalid molecule_id")
    if int(molecules["molecule_id"][row]) != int(molecule_id):
        raise ValueError("Timed-output molecule IDs are not sequential")
    atom_start = int(molecules["atom_offsets"][row])
    atom_stop = int(molecules["atom_offsets"][row + 1])
    bond_start = int(molecules["bond_offsets"][row])
    bond_stop = int(molecules["bond_offsets"][row + 1])
    atom_ids = tuple(
        int(value) for value in molecules["atom_ids"][atom_start:atom_stop]
    )
    bonds = tuple(
        (int(pair[0]), int(pair[1]), int(order))
        for pair, order in zip(
            molecules["bond_atoms"][bond_start:bond_stop],
            molecules["bond_order"][bond_start:bond_stop],
        )
    )
    species_id = int(molecules["species_id"][row])
    species = _decode_hdf5_string(handle["species/name"][species_id - 1])
    return TransitionParticipant(
        side=side,
        molecule_id=int(molecule_id),
        species=species,
        atom_ids=atom_ids,
        bonds=bonds,
    )


def iter_transition_evidence(
    filename: str | Path,
) -> Iterator[TransitionEvidence]:
    """Yield molecule-instance and bond-change evidence in transition order."""
    with _open_timed_output(filename) as handle:
        if "transition_evidence" not in handle:
            raise ValueError(
                "Timed-output HDF5 file does not contain transition evidence"
            )
        reaction_types = handle["reaction_types"]
        evidence = handle["transition_evidence"]
        for transition_index, (block_start, block_length) in enumerate(
            zip(evidence["block_start"], evidence["block_length"])
        ):
            event_start = int(block_start)
            event_stop = event_start + int(block_length)
            for event_row in range(event_start, event_stop):
                stored_transition = int(evidence["transition_index"][event_row])
                if stored_transition != transition_index:
                    raise ValueError(
                        "Transition evidence block disagrees with transition_index"
                    )
                reaction_id = int(evidence["reaction_id"][event_row])
                reactant = _decode_hdf5_string(
                    reaction_types["reactant"][reaction_id - 1]
                )
                product = _decode_hdf5_string(
                    reaction_types["product"][reaction_id - 1]
                )
                participant_start = int(evidence["participant_offsets"][event_row])
                participant_stop = int(evidence["participant_offsets"][event_row + 1])
                participants = tuple(
                    _read_transition_participant(
                        handle,
                        int(molecule_id),
                        "reactant" if int(side) == 0 else "product",
                    )
                    for molecule_id, side in zip(
                        evidence["participant_molecule_id"][
                            participant_start:participant_stop
                        ],
                        evidence["participant_side"][
                            participant_start:participant_stop
                        ],
                    )
                )
                change_start = int(evidence["bond_change_offsets"][event_row])
                change_stop = int(evidence["bond_change_offsets"][event_row + 1])
                bond_changes = tuple(
                    BondChange(
                        atom1=int(pair[0]),
                        atom2=int(pair[1]),
                        before_order=int(before_order),
                        after_order=int(after_order),
                    )
                    for pair, before_order, after_order in zip(
                        evidence["bond_atoms"][change_start:change_stop],
                        evidence["before_order"][change_start:change_stop],
                        evidence["after_order"][change_start:change_stop],
                    )
                )
                yield TransitionEvidence(
                    transition_index=transition_index,
                    reactant=reactant,
                    product=product,
                    participants=participants,
                    bond_changes=bond_changes,
                )


def read_species(
    specfile: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read species from the species file (ends with .species).

    For accuracy, HMM filter should be disabled.

    Parameters
    ----------
    specfile : str or Path
        The species file.

    Returns
    -------
    step_idx : np.ndarray
        The index of the step.
    n_species : Dict[str, np.ndarray]
        The number of species in each step. The dict key is the species SMILES.

    Examples
    --------
    Plot the number of methane in each step.

    >>> from reacnetgenerator.tools import read_species
    >>> import matplotlib.pyplot as plt
    >>> step_idx, n_species = read_species('methane.species')
    >>> plt.plot(step_idx, n_species['[H][C]([H])([H])[H]'])
    >>> plt.savefig("methane.svg")
    """
    step_idx = []
    n_species = defaultdict(lambda: defaultdict(int))
    with open(specfile) as f:
        ii = -1
        for ii, line in enumerate(f):
            s = line.split()
            step_idx.append(int(s[1].strip(":")))
            for ss, nn in zip(s[2::2], [int(x) for x in s[3::2]]):
                n_species[ss][ii] = nn
        else:
            nsteps = ii + 1
    n_species2 = {}
    for ss in n_species:
        n_species2[ss] = np.array(
            [n_species[ss][ii] for ii in range(nsteps)], dtype=int
        )
    return np.array(step_idx, dtype=int), n_species2


def read_reactions(reacfile: str | Path) -> list[tuple[int, Counter, str]]:
    """Read reactions from the reactions file (ends with .reaction or .reactionsabcd).

    For accuracy, HMM filter should be disabled.

    Parameters
    ----------
    reacfile : str or Path
        The reactions file.

    Returns
    -------
    occs : List[Tuple[int, Counter, str]]
        The number of occurences of each reaction. The tuple is (occurence, counter_reactants, reaction).
    """
    occs = []
    with open(reacfile) as f:
        for line in f:
            s = line.split()
            occs.append((int(s[0]), Counter(s[1].split("->")[0].split("+")), s[1]))
    return occs


def calculate_rate(
    specfile: str | Path,
    reacfile: str | Path,
    cell: np.ndarray,
    timestep: float,
) -> dict[str, float]:
    """Calculate the rate constant of each reaction.

    The rate constants are calculated by the method developed in [1]_.
    The time interval of the trajectory is assumed to be uniform.

    Parameters
    ----------
    specfile : str
        The species file.
    reacfile : str
        The reactions file.
    cell : np.ndarray
        The cell with the shape (3, 3). Unit: Angstrom.
    timestep : float
        The time step. Unit: femtosecond.

    Returns
    -------
    rates : Dict[str, float]
        The rate of each reaction. The dict key is the reaction SMILES.
        The value is in unit of [(cm^3/mol)^(n-1)s^(-1)], where n is the reaction order.

    References
    ----------
    .. [1] Yanze Wu, Huai Sun, Liang Wu, Joshua D. Deetz, Extracting the mechanisms
       and kinetic models of complex reactions from atomistic simulation data, J.
       Comput. Chem. 40, 16, 1586-1592.

    Examples
    --------
    >>> cell = np.eye(3) * 3.7601e1 # in unit Angstrom
    >>> timestep = 0.1 # in unit fs
    >>> rates = calculate_rate('methane.species', 'methane.reactionabcd', cell, timestep)
    """
    ase_cell = ase.geometry.Cell(cell)

    timestep *= 10**-15  # fs to s
    # N, step_tot =read_species(specfile)
    step_idx, n_species = read_species(specfile)
    occs = read_reactions(reacfile)

    # time interval between two frames
    time_int = (step_idx[1] - step_idx[0]) * timestep
    # volume of the cell
    volume = ase_cell.volume
    volume *= 10**-24  # Ang^3 to cm^3
    volume_times_na = volume * ase.units.mol  # V * NA

    rates = {}
    for occ, reacts, reactions in occs:
        # k = occ_tot / ( V * time_tot * c_tot )
        # c_tot = N_tot / (V * NA)
        n_react = np.array([n_species[kk] for kk in reacts.keys()])
        nu = np.array(list(reacts.values()))
        c_po = np.power(
            n_react / volume_times_na,
            np.repeat(nu, n_react.shape[1]).reshape(n_react.shape),
        )
        c_tot = np.sum(np.prod(c_po, axis=0))
        k = occ / (volume_times_na * time_int * c_tot)
        rates[reactions] = k
    return rates
