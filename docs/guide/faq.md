# FAQ

## Can ReacNetGenerator process my trajectory?

Generally ReacNetGenerator has no limitation on any specific elements.

When perceiving bond orders from atomic coordinates, ReacNetGenerator will call [Open Babel][openbabel] which reads [element parameters](https://github.com/openbabel/openbabel/blob/2f34bda337d7ddefa8f2bebfc23931a63e45241f/src/elementtable.h).{cite:p}`O'Boyle_JCheminform_2011_v3_p33`
The parameters may not fit the system.
If you have new ideas about parameters, you can report to [Open Babel][openbabel] and recompile Open Babel from the new source code.

When processing a ReaxFF bond file, bond orders are directly provided by ReaxFF with decimal rounding.{cite:p}`Aktulga_ParallelComputing_2012_v38_p245`
The accuracy of the bond orders depends on the accuracy of the force field.

You can refer the list of [publications driven by ReacNetGenerator](https://njzjz.win/reacnetgenerator/) to see other researchers' applications.

## Out of memory (OOM)

When processing a large trajectory, you may get different OOM errors such as `Memory Error`, or subprocesses are directly killed by the system with `broken pipe` thrown.
If you have a machine that has more memory, just use it.
Otherwise, try to reduce the size of the trajectory or split the trajectory into multiple files or increase the number given by `--stepinterval`.
It is also helpful to recuding the number of processes.

Time-resolved molecule and reaction output uses a `.timeline.h5` HDF5 file.
Molecule lifetimes are stored as ranges and reaction events are aggregated on
disk, so enabling these outputs does not materialize one Python or CSV row per
occurrence. The HDF5 raw-data chunk cache is bounded to 128 MiB by default and
can be adjusted with
`--timed-output-cache-mib`.

Step 3 stores its atom-by-frame working matrices in temporary memory-mapped
files. Workers attach those mappings once and receive only atom or transition
indices, with one index per multiprocessing chunk. This bounds IPC payloads and
keeps the complete matrices out of each worker's resident Python heap. It does
increase temporary disk use, so the temporary filesystem must have enough free
space.

If you are using a Windows OS, it's known that the program may consume large memory through multiprocessing.
In this situation, it's suggested to use [Windows Subsystem Linux (WSL)](https://docs.microsoft.com/windows/wsl).

[openbabel]: https://github.com/openbabel/openbabel
