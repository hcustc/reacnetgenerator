# Generated files

The following files are reported:

## Web page

suffix: `.html`

The web page including analysis.
<a href="../report.html?jdata=https%3A%2F%2Fgist.githubusercontent.com%2Fnjzjz%2Fe9a4b42ceb7d2c3c7ada189f38708bf3%2Fraw%2F83d01b9ab1780b0ad2d1e7f934e61fa113cb0f9f%2Fmethane.json" target="_blank">Here</a> is an example.
You can open it using a modern browser.
Note that $\ce{A + B -> C + D}$ information may be not accurate when [HMM](hmm.md) is enabled.

## Data file

suffix: `.json`

The JSON file storing necessary data for the web page.
You can load it through the <a href="../report.html" target="_blank">web page loader</a>.

## Species file

suffix: `.species`

This text file stores the number of each species in each time step.
One can read this file through the Python method {meth}`reacnetgenerator.tools.read_species <reacnetgenerator.tools.read_species>`.
Note that when [HMM](hmm.md) is enabled, information in this file may not be accurate.

## Molecule file

suffix: `.moname`

This file contains information of each molecule.
In each line, the first column is its SMILES.
The second column is the atomic index (starts from 0) of atoms.
The last column shows all the bonds in the molecule.
This file always keeps the historical three-column format.

## Time-resolved HDF5 file

suffix: `.timeline.h5`

This optional HDF5 file is written when
`--show-molecule-time`, `--molecule-frame`, `--molecule-timestep`, or
`--reaction-event` is enabled. Use `--timed-output FILE` to select a different
path and `--timed-output-cache-mib` to set the bounded HDF5 raw-data chunk
cache.

The file stores source-file provenance in `sources`, analyzed-frame metadata in
`frames`, molecule definitions in `molecules`, and closed
existence intervals in `molecule_ranges`. A molecule present from analyzed
frame 10 through frame 1000 is therefore stored as one range instead of 991
repeated rows. The optional frame and timestep filters restrict these stored
ranges.

`frame` is the zero-based continuous analyzed-frame index after concatenating
the input files and applying the global `--stepinterval`. `source_frame` is the
zero-based frame index in its original input file before applying that
interval, while `timestep` is the trajectory's original timestep value and may
repeat between input files.

Reaction definitions and totals are stored in `reaction_types`, and aggregated
occurrences are stored in `reaction_events`. `transition_index = f` means the
transition from analyzed frame `f` to `f + 1`. Worker results may arrive out of
order; `block_start` and `block_length`, indexed by transition, point to each
transition's compact event block and preserve logical time order without
buffering all worker results in memory.

Only a file whose root `status` attribute is `complete` is published at the
formal path. During construction, a job-specific temporary HDF5 file is kept in
the same directory. Failed builds can leave that `.tmp` artifact for diagnosis,
but never replace the formal file.

The old `.molecules.csv` and `.reactionevent.csv` outputs are no longer
available. Python callers should replace `moleculetimelinefilename` and
`reactioneventfilename` with `timedoutputfilename`.

The normalized records can be read lazily:

```python
from reacnetgenerator.tools import (
    iter_molecule_timeline,
    iter_reaction_events,
    read_timed_output_metadata,
)

metadata = read_timed_output_metadata("bonds.reaxc.timeline.h5")
for timestep, species, atom_ids, bond_ids in iter_molecule_timeline(
    "bonds.reaxc.timeline.h5"
):
    ...

for transition_index, reactant, product in iter_reaction_events(
    "bonds.reaxc.timeline.h5"
):
    ...
```

## Route file

suffix: `.route`

This file contains the route of each atom.
It shows which species an atom is inside thorugh the whole trajectory in the following format:

```
Atom {idx}: {time} {SMILES} -> {time} {SMILES} -> ...
```

## Reaction files

suffix: `.reaction`, `.reactionabcd`

`.reaction` shows the frequency of the reaction $\ce{A -> B}$ while `.reactionabcd` shows the frequency of the reaction $\ce{A + B -> C + D}$.
One can read these files through the Python method {meth}`reacnetgenerator.tools.read_reactions <reacnetgenerator.tools.read_reactions>`.
Note that $\ce{A + B -> C + D}$ information may be not accurate when [HMM](hmm.md) is enabled.

## Reaction matrix file

suffix: `.table`

This file shows the reaction matrix.
To save space, only first top 100 species are printed.
You can process [reaction files](#reaction-files) manually for other information.

## Reaction network

suffix: `.svg`

A SVG file to show reaction networks.
