#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later

set -euo pipefail

environment_prefix="${1:?usage: run-remote-reacnetgenerator-validation.sh ENV_PREFIX INPUT REPEAT [BASELINE]}"
input_file="${2:?input trajectory is required}"
repeat_count="${3:?repeat count is required}"
baseline_manifest="${4:-}"

[[ -x "$environment_prefix/bin/python" ]] || {
    echo "remote environment is missing: $environment_prefix" >&2
    exit 2
}
[[ -f "$input_file" ]] || { echo "input is missing: $input_file" >&2; exit 2; }
[[ "$repeat_count" =~ ^[1-9][0-9]*$ ]] || {
    echo "repeat count must be positive" >&2
    exit 2
}

python_executable="$environment_prefix/bin/python"
installed_dps="$($python_executable -c 'from reacnetgenerator import dps; print(dps.__file__)')"
cp -- "$installed_dps" source/reacnetgenerator/

find source/reacnetgenerator source/scripts \
    -type f \
    -print0 \
    | sort -z \
    | while IFS= read -r -d '' source_file; do
        sha256sum -- "$source_file"
    done >source.sha256

export PYTHONPATH="$PWD/source"
export RNG_PYTHON="$python_executable"
export RNG_MONITOR_INTERVAL_SECONDS="${RNG_MONITOR_INTERVAL_SECONDS:-5}"
export SLURM_TMPDIR="${SLURM_TMPDIR:-$PWD/node-tmp}"
mkdir -p -- "$SLURM_TMPDIR"

{
    echo "utc_task_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
    echo "environment_prefix=$environment_prefix"
    echo "python=$python_executable"
    echo "installed_dps=$installed_dps"
    echo "input_file=$input_file"
    echo "repeat_count=$repeat_count"
    sha256sum -- "$input_file"
    sha256sum -- source.sha256
    "$python_executable" -c 'import h5py, numpy, reacnetgenerator, scipy; from openbabel import openbabel as ob; from reacnetgenerator import dps; print(f"source_module={reacnetgenerator.__file__}"); print(f"active_dps={dps.__file__}"); print(f"active_version={reacnetgenerator.__version__}"); print(f"openbabel_release={ob.OBReleaseVersion()}"); print(f"h5py={h5py.__version__}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
} >environment.txt

input_arguments=()
for ((index = 0; index < repeat_count; index++)); do
    input_arguments+=(--input "$input_file")
done

baseline_arguments=()
if [[ -n "$baseline_manifest" ]]; then
    [[ -f "$baseline_manifest" ]] || {
        echo "baseline manifest is missing: $baseline_manifest" >&2
        exit 2
    }
    baseline_arguments=(--baseline "$baseline_manifest")
fi

bash source/scripts/slurm-production-validation.sh \
    "${input_arguments[@]}" \
    --output-root results \
    --atoms C,H,O \
    --type dump \
    --nproc "${SLURM_CPUS_PER_TASK:-48}" \
    "${baseline_arguments[@]}" \
    -- --nohmm

candidate_manifest="$(find results -mindepth 2 -maxdepth 2 -name candidate.timeline.manifest.json -print -quit)"
"$python_executable" - "$candidate_manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
print(f"manifest_status={manifest['status']}")
print(f"semantic_fingerprint={manifest['semantic_fingerprint']}")
print(f"counts={manifest['counts']}")
print(f"performance_seconds={manifest['performance_seconds']}")
PY

# DPDispatcher's compressed download dereferences result-directory symlinks.
# Remove only the generated input links after validation so a repeated-input
# test does not copy the same large trajectory back once per logical source.
removed_input_links=0
while IFS= read -r -d '' input_link; do
    unlink -- "$input_link"
    removed_input_links=$((removed_input_links + 1))
done < <(
    find results \
        -mindepth 2 \
        -maxdepth 2 \
        -type l \
        -name 'trajectory.*.input' \
        -print0
)
echo "removed_result_input_links=$removed_input_links" >>environment.txt
