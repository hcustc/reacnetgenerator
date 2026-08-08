#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Submit one reproducible Slurm task through DPDispatcher over SSH."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dpdispatcher import Machine, Resources, Submission, Task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--task-work-path", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--forward", action="append", default=[])
    parser.add_argument("--backward", action="append", default=[])
    parser.add_argument("--cpus", type=int, default=48)
    parser.add_argument("--memory-gib", type=int, default=220)
    parser.add_argument("--time-limit", default="06:00:00")
    parser.add_argument("--job-name", default="rng-validation")
    parser.add_argument("--queue", default="main")
    parser.add_argument("--hostname", default="n3.jinzhezeng.group")
    parser.add_argument("--username", default="chuang")
    parser.add_argument(
        "--key-file", type=Path, default=Path("~/.ssh/id_ed25519_chuang")
    )
    parser.add_argument("--remote-root", default="/home/chuang/dpdispatcher_works")
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Validate local inputs, submit the task, and wait for its outputs."""
    arguments = _parser().parse_args()
    local_root = arguments.local_root.expanduser().resolve()
    key_file = arguments.key_file.expanduser().resolve()
    task_path = Path(arguments.task_work_path)

    if not local_root.is_dir():
        raise SystemExit(f"local root does not exist: {local_root}")
    if task_path.is_absolute() or ".." in task_path.parts:
        raise SystemExit("task work path must be relative and must not contain '..'")
    local_task = local_root / task_path
    if not local_task.is_dir():
        raise SystemExit(f"local task directory does not exist: {local_task}")
    if not key_file.is_file():
        raise SystemExit(f"SSH key does not exist: {key_file}")
    if not 1 <= arguments.cpus <= 48:
        raise SystemExit("n3 supports between 1 and 48 CPUs per task")
    if not 1 <= arguments.memory_gib <= 235:
        raise SystemExit("memory request must be between 1 and 235 GiB")
    if not re.fullmatch(r"(?:\d+-)?\d{1,2}:\d{2}:\d{2}", arguments.time_limit):
        raise SystemExit("time limit must use [days-]HH:MM:SS")
    if arguments.check_interval <= 0:
        raise SystemExit("check interval must be positive")
    if arguments.retry_count < 0:
        raise SystemExit("retry count must not be negative")
    for relative_name in arguments.forward:
        if not (local_task / relative_name).exists():
            raise SystemExit(f"forward path does not exist: {relative_name}")

    machine = Machine.load_from_dict(
        {
            "batch_type": "Slurm",
            "context_type": "SSHContext",
            "local_root": str(local_root),
            "remote_root": arguments.remote_root,
            "remote_profile": {
                "hostname": arguments.hostname,
                "username": arguments.username,
                "port": 22,
                "key_filename": str(key_file),
                "look_for_keys": False,
                "timeout": 30,
                "tar_compress": True,
            },
            "retry_count": arguments.retry_count,
            "clean_asynchronously": False,
        }
    )
    resources = Resources(
        number_node=1,
        cpu_per_node=arguments.cpus,
        gpu_per_node=0,
        queue_name=arguments.queue,
        group_size=1,
        custom_flags=[
            "#SBATCH --ntasks=1",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={arguments.cpus}",
            f"#SBATCH --mem={arguments.memory_gib}G",
            f"#SBATCH --time={arguments.time_limit}",
            f"#SBATCH --job-name={arguments.job_name}",
        ],
        custom_gpu_line="# GPU not requested",
    )
    backward_files = list(dict.fromkeys([*arguments.backward, "task.out", "task.err"]))
    task = Task(
        command=arguments.command,
        task_work_path=arguments.task_work_path,
        forward_files=arguments.forward,
        backward_files=backward_files,
        outlog="task.out",
        errlog="task.err",
    )
    submission = Submission(
        work_base=".",
        machine=machine,
        resources=resources,
        task_list=[task],
    )
    print(
        json.dumps(
            {
                "local_root": str(local_root),
                "task_work_path": arguments.task_work_path,
                "hostname": arguments.hostname,
                "remote_root": arguments.remote_root,
                "cpus": arguments.cpus,
                "memory_gib": arguments.memory_gib,
                "time_limit": arguments.time_limit,
                "job_name": arguments.job_name,
                "retry_count": arguments.retry_count,
                "dry_run": arguments.dry_run,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    submission.run_submission(
        dry_run=arguments.dry_run,
        clean=False,
        check_interval=arguments.check_interval,
    )
    print("DPDispatcher submission completed and backward files downloaded.")


if __name__ == "__main__":
    main()
