#!/usr/bin/env bash
#SBATCH --job-name=rng-production-validation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Run one isolated full-trajectory ReacNetGenerator validation inside Slurm.

Usage:
  slurm-production-validation.sh \
      --input TRAJECTORY [--input TRAJECTORY ...] \
      --output-root DIRECTORY \
      --atoms C,H,O \
      [--type dump] [--nproc 48] [--baseline MANIFEST] \
      [--timed-output-cache-mib 1] \
      [-- REACNETGENERATOR_ARGUMENT ...]

The script always enables --show-molecule-time and --reaction-event, writes all
outputs below a new run-<job-id> directory, and validates candidate.timeline.h5.
Set RNG_PYTHON to the Python executable containing this checkout/install.
EOF
}

inputs=()
output_root=""
atom_csv=""
input_type="dump"
nproc="${SLURM_CPUS_PER_TASK:-1}"
baseline=""
timed_output_cache_mib="1"
extra_args=()

while (($#)); do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || { echo "--input requires a value" >&2; exit 2; }
            inputs+=("$2")
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
            output_root="$2"
            shift 2
            ;;
        --atoms)
            [[ $# -ge 2 ]] || { echo "--atoms requires a value" >&2; exit 2; }
            atom_csv="$2"
            shift 2
            ;;
        --type)
            [[ $# -ge 2 ]] || { echo "--type requires a value" >&2; exit 2; }
            input_type="$2"
            shift 2
            ;;
        --nproc)
            [[ $# -ge 2 ]] || { echo "--nproc requires a value" >&2; exit 2; }
            nproc="$2"
            shift 2
            ;;
        --baseline)
            [[ $# -ge 2 ]] || { echo "--baseline requires a value" >&2; exit 2; }
            baseline="$2"
            shift 2
            ;;
        --timed-output-cache-mib)
            [[ $# -ge 2 ]] || {
                echo "--timed-output-cache-mib requires a value" >&2
                exit 2
            }
            timed_output_cache_mib="$2"
            shift 2
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown wrapper argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ ${#inputs[@]} -gt 0 ]] || { echo "At least one --input is required" >&2; exit 2; }
[[ -n "$output_root" ]] || { echo "--output-root is required" >&2; exit 2; }
[[ -n "$atom_csv" ]] || { echo "--atoms is required" >&2; exit 2; }
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] || { echo "--nproc must be positive" >&2; exit 2; }
[[ "$timed_output_cache_mib" =~ ^[1-9][0-9]*$ ]] || {
    echo "--timed-output-cache-mib must be positive" >&2
    exit 2
}

case "$input_type" in
    bond|lammpsbondfile|dump|lammpsdumpfile|xyz|extxyz) ;;
    *) echo "Unsupported --type: $input_type" >&2; exit 2 ;;
esac

IFS=',' read -r -a atoms <<<"$atom_csv"
[[ ${#atoms[@]} -gt 0 ]] || { echo "--atoms must not be empty" >&2; exit 2; }
for atom in "${atoms[@]}"; do
    [[ "$atom" =~ ^[A-Za-z][A-Za-z0-9]*$ ]] || {
        echo "Invalid atom name in --atoms: $atom" >&2
        exit 2
    }
done

for argument in "${extra_args[@]}"; do
    case "$argument" in
        -i|--inputfilename|--inputfilename=*|-a|--atomname|--atomname=*|\
        -n|-np|--nproc|--nproc=*|-t|--type|--type=*|--dump|\
        --show-molecule-time|--reaction-event|--timed-output|\
        --timed-output=*|--timed-output-cache-mib|--timed-output-cache-mib=*)
            echo "Wrapper-managed argument must not follow --: $argument" >&2
            exit 2
            ;;
    esac
done

absolute_path() {
    local path="$1"
    local directory
    local basename
    directory="$(dirname -- "$path")"
    basename="$(basename -- "$path")"
    (cd "$directory" && printf '%s/%s\n' "$(pwd -P)" "$basename")
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1"
    else
        shasum -a 256 -- "$1"
    fi
}

resolved_inputs=()
for input in "${inputs[@]}"; do
    [[ -f "$input" ]] || { echo "Input does not exist: $input" >&2; exit 2; }
    resolved_inputs+=("$(absolute_path "$input")")
done

if [[ -n "$baseline" ]]; then
    [[ -f "$baseline" ]] || { echo "Baseline does not exist: $baseline" >&2; exit 2; }
    baseline="$(absolute_path "$baseline")"
fi

mkdir -p -- "$output_root"
output_root="$(cd "$output_root" && pwd -P)"
run_label="${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
run_dir="$output_root/run-$run_label"
if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing run directory: $run_dir" >&2
    exit 2
fi
mkdir -- "$run_dir"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
rng_python="${RNG_PYTHON:-python}"
rng_python_command="$(command -v "$rng_python" 2>/dev/null)" || {
    echo "Python executable not found: $rng_python" >&2
    exit 2
}
if [[ "$rng_python_command" = /* ]]; then
    rng_python="$rng_python_command"
else
    rng_python="$(absolute_path "$rng_python_command")"
fi

job_tmp_parent="${SLURM_TMPDIR:-$run_dir/tmp}"
job_tmp_dir="$job_tmp_parent/reacnetgenerator-$run_label"
mkdir -p -- "$job_tmp_dir"
export TMPDIR="$job_tmp_dir"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$job_tmp_dir/matplotlib}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

local_inputs=()
index=0
for input in "${resolved_inputs[@]}"; do
    printf -v local_name 'trajectory.%04d.input' "$index"
    ln -s -- "$input" "$run_dir/$local_name"
    local_inputs+=("$local_name")
    index=$((index + 1))
done

{
    echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "run_directory=$run_dir"
    echo "temporary_directory=$job_tmp_dir"
    echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
    echo "slurm_job_name=${SLURM_JOB_NAME:-unset}"
    echo "slurm_node_list=${SLURM_JOB_NODELIST:-unset}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
    echo "nproc=$nproc"
    echo "input_type=$input_type"
    echo "atoms=$atom_csv"
    echo "timed_output_cache_mib=$timed_output_cache_mib"
    echo "omp_num_threads=$OMP_NUM_THREADS"
    echo "openblas_num_threads=$OPENBLAS_NUM_THREADS"
    echo "mkl_num_threads=$MKL_NUM_THREADS"
    git -C "$repo_root" rev-parse --verify HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
    "$rng_python" -c 'import importlib.metadata as metadata, importlib.util as util, os; import reacnetgenerator as rng; affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unavailable"; spec = util.find_spec("reacnetgenerator"); print("python_affinity=%s" % affinity); print("reacnetgenerator_version=%s" % rng.__version__); print("installed_distribution_version=%s" % metadata.version("reacnetgenerator")); print("reacnetgenerator_module=%s" % (spec.origin if spec else "not-found"))'
} >"$run_dir/metadata.txt"

git -C "$repo_root" status --short >"$run_dir/git-status.txt" 2>&1 || true
for input in "${resolved_inputs[@]}"; do
    hash_file "$input"
done >"$run_dir/input.sha256"

timed_output="candidate.timeline.h5"
command_line=(
    "$rng_python" -m reacnetgenerator
    --type "$input_type"
    -i "${local_inputs[@]}"
    -a "${atoms[@]}"
    -n "$nproc"
    --show-molecule-time
    --reaction-event
    --timed-output "$timed_output"
    --timed-output-cache-mib "$timed_output_cache_mib"
    "${extra_args[@]}"
)

{
    printf 'cd %q\n' "$run_dir"
    printf '%q ' "${command_line[@]}"
    printf '\n'
} >"$run_dir/command.txt"

cat >"$run_dir/post-job-sacct-command.txt" <<EOF
sacct -j ${SLURM_JOB_ID:-JOB_ID} --units=G --parsable2 --format=JobIDRaw,State,ElapsedRaw,AllocCPUS,TotalCPU,CPUTimeRAW,MaxRSS,AveRSS,MaxVMSize,ExitCode
EOF

monitor_interval="${RNG_MONITOR_INTERVAL_SECONDS:-60}"
[[ "$monitor_interval" =~ ^[1-9][0-9]*$ ]] || {
    echo "RNG_MONITOR_INTERVAL_SECONDS must be positive" >&2
    exit 2
}
monitor_pid=""
slurm_monitor_pid=""
process_monitor_pid=""
wrapper_pid="$$"
monitor_tmp_usage() {
    trap 'exit 0' TERM INT
    while true; do
        printf '%s\t' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        du -sk -- "$job_tmp_dir" 2>/dev/null | awk '{print $1}'
        sleep "$monitor_interval" &
        wait $!
    done
}

monitor_slurm_usage() {
    trap 'exit 0' TERM INT
    local sample
    while true; do
        sample="$(sstat \
            --jobs="${SLURM_JOB_ID}.batch" \
            --noheader \
            --parsable2 \
            --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize \
            2>/dev/null || true)"
        printf '%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "${sample:-unavailable}"
        sleep "$monitor_interval" &
        wait $!
    done
}

monitor_process_tree_usage() {
    trap 'exit 0' TERM INT
    printf 'utc\tprocess_count\ttotal_rss_kib\ttotal_vsz_kib\tmax_process_rss_kib\n'
    while true; do
        printf '%s\t' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        ps -eo pid=,ppid=,rss=,vsz= | awk -v root="$wrapper_pid" '
            {
                ids[++count] = $1
                parent[$1] = $2
                rss[$1] = $3
                vsz[$1] = $4
            }
            END {
                included[root] = 1
                changed = 1
                while (changed) {
                    changed = 0
                    for (item_index = 1; item_index <= count; item_index++) {
                        pid = ids[item_index]
                        if (!included[pid] && included[parent[pid]]) {
                            included[pid] = 1
                            changed = 1
                        }
                    }
                }
                for (item_index = 1; item_index <= count; item_index++) {
                    pid = ids[item_index]
                    if (included[pid]) {
                        processes++
                        total_rss += rss[pid]
                        total_vsz += vsz[pid]
                        if (rss[pid] > max_rss) {
                            max_rss = rss[pid]
                        }
                    }
                }
                printf "%d\t%d\t%d\t%d\n", processes, total_rss, total_vsz, max_rss
            }
        '
        sleep "$monitor_interval" &
        wait $!
    done
}

stop_monitor() {
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    monitor_pid=""
    if [[ -n "$slurm_monitor_pid" ]] && kill -0 "$slurm_monitor_pid" 2>/dev/null; then
        kill "$slurm_monitor_pid" 2>/dev/null || true
        wait "$slurm_monitor_pid" 2>/dev/null || true
    fi
    slurm_monitor_pid=""
    if [[ -n "$process_monitor_pid" ]] && kill -0 "$process_monitor_pid" 2>/dev/null; then
        kill "$process_monitor_pid" 2>/dev/null || true
        wait "$process_monitor_pid" 2>/dev/null || true
    fi
    process_monitor_pid=""
}

finalize() {
    local exit_status=$?
    set +e
    stop_monitor
    {
        echo "wrapper_exit_status=$exit_status"
        echo "utc_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "run_directory=$run_dir"
        du -sk -- "$run_dir"
        du -sk -- "$job_tmp_dir"
    } >"$run_dir/final-status.txt" 2>&1
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        scontrol show job "$SLURM_JOB_ID" >"$run_dir/slurm-job-end.txt" 2>&1 || true
    fi
    (
        cd "$run_dir" || exit
        find . -maxdepth 1 -type f ! -name 'output.sha256.new' -print0 \
            | sort -z \
            | while IFS= read -r -d '' file; do hash_file "$file"; done \
            >output.sha256.new
        mv -- output.sha256.new output.sha256
    )
    trap - EXIT
    exit "$exit_status"
}
trap finalize EXIT

monitor_tmp_usage >"$run_dir/tmp-usage-kib.tsv" &
monitor_pid=$!
monitor_process_tree_usage >"$run_dir/process-stats.tsv" &
process_monitor_pid=$!
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sstat >/dev/null 2>&1; then
    monitor_slurm_usage >"$run_dir/slurm-stats.tsv" &
    slurm_monitor_pid=$!
fi
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show job "$SLURM_JOB_ID" >"$run_dir/slurm-job-start.txt" 2>&1 || true
fi

cd "$run_dir"
set +e
raw_run_status="$run_dir/run-status.raw"
timed_command=(
    /bin/bash -c
    'status_file=$1; shift; "$@"; child_status=$?; printf "%s\n" "$child_status" >"$status_file"; exit "$child_status"'
    bash "$raw_run_status" "${command_line[@]}"
)
resource_time_status="not-run"
if [[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU time'; then
    /usr/bin/time -v -o resource-time.txt "${timed_command[@]}" 2>&1 | tee run.log
    resource_time_status=${PIPESTATUS[0]}
elif [[ -x /usr/bin/time && "$(uname -s)" == "Darwin" ]]; then
    /usr/bin/time -l -o resource-time.txt "${timed_command[@]}" 2>&1 | tee run.log
    resource_time_status=${PIPESTATUS[0]}
else
    "${command_line[@]}" 2>&1 | tee run.log
    run_status=${PIPESTATUS[0]}
fi
if [[ -s "$raw_run_status" ]]; then
    read -r run_status <"$raw_run_status"
elif [[ "$resource_time_status" =~ ^[0-9]+$ ]]; then
    run_status="$resource_time_status"
fi
set -e
{
    printf 'reacnetgenerator_exit_status=%s\n' "$run_status"
    printf 'resource_time_exit_status=%s\n' "$resource_time_status"
} >run-status.txt
if ((run_status != 0)); then
    exit "$run_status"
fi

validation_command=(
    "$rng_python" -m reacnetgenerator.timedoutputcheck
    "$timed_output"
    --output candidate.timeline.manifest.json
)
if [[ -n "$baseline" ]]; then
    validation_command+=(--baseline "$baseline")
fi
printf '%q ' "${validation_command[@]}" >validation-command.txt
printf '\n' >>validation-command.txt
"${validation_command[@]}" 2>&1 | tee validation.log
