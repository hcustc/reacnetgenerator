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
occurrence. The sequential writer uses a 1 MiB HDF5 raw-data chunk cache by
default, avoiding retention of chunks that will not be read again, and the cache
can be adjusted with
`--timed-output-cache-mib`.

Step 3 stores its atom-by-frame working matrices in temporary memory-mapped
files. Workers attach those mappings once and receive only atom or transition
indices, with one index per multiprocessing chunk. This bounds IPC payloads and
keeps the complete matrices out of each worker's resident Python heap. It does
increase temporary disk use, so the temporary filesystem must have enough free
space.

The SMILES, atom-route, and reaction stages may automatically use fewer workers
when the measured structure, timeline, or observed reaction work is too small to
amortize multiprocessing. The selected worker count is written to the log. This
can lower the reported CPU utilization while still reducing wall time, memory,
IPC, and total core-hours; use that log to request fewer CPUs for subsequent runs
when appropriate.

For inexpensive molecule records on systems using the `fork` start method, the
batched SMILES workers also return the structures they already decoded, but only
when the HMM/filter pass reports that every compressed structure record is at
most 64 KiB. The parent then reads only the timeline payload, instead of
decompressing the atom and bond fields a second time. Serial runs,
`spawn`/`forkserver`, an unknown maximum, or any larger record keep the compact
name-only worker result to avoid enlarging IPC and memory use where the tradeoff
has not been validated. The selected behavior and maximum record size are
reported in the log.

Atom-route analysis also records which adjacent frames contain at least one
molecule-ID change. Reaction analysis scans only those active transitions; a
fully stable trajectory can therefore finish the reaction stage without reading
every atom column again. The index uses one temporary byte per transition and
does not send per-atom change arrays through multiprocessing queues. The log and
timed-output HDF5 attributes report active versus total transition counts.

At startup, the log also reports the requested `nproc`, CPUs visible to the
process, logical CPU count, compact CPU-affinity ranges, and
`SLURM_CPUS_PER_TASK`. A warning is emitted when the requested process count
exceeds the visible CPUs or when the Slurm value disagrees with process
affinity. These warnings do not change an explicitly requested `nproc`; they
identify CPU binding, cgroup, or accidental oversubscription before a long run.

If you are using a Windows OS, it's known that the program may consume large memory through multiprocessing.
In this situation, it's suggested to use [Windows Subsystem Linux (WSL)](https://docs.microsoft.com/windows/wsl).

[openbabel]: https://github.com/openbabel/openbabel
