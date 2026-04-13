# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict


@dataclass
class StoolArgs:
    config: Any = None
    launcher: str = "sbatch"  # Can be sbatch or bash if already in salloc
    script: str = "apps.main.train"  # The script to run.
    copy_code: bool = True  # Wether to copy code to dump dir
    dirs_exists_ok: bool = (
        False  # Wether to copy new code and config and run regardless that dir exists
    )
    override: bool = False  # Wether to delete dump dir and restart
    nodes: int = -1  # The number of nodes to run the job on.
    ngpu: int = 8  # The number of GPUs required per node.
    ncpu: int = 16  # The number of CPUs allocated per GPU.
    mem: str = ""  # The amount of memory to allocate.
    anaconda: str = "default"  # The path to the anaconda environment.
    constraint: str = ""  # The constraint on the nodes.
    exclude: str = ""  # The nodes to exclude.
    time: int = -1  # The time limit of the job (in minutes).
    account: str = ""
    qos: str = ""
    partition: str = "learn"
    stdout: bool = False


SBATCH_COMMAND = """#!/bin/bash

{exclude}
{qos}
{account}
{constraint}
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-gpu={ncpu}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --mem={mem}

#SBATCH --output={dump_dir}/logs/%j/%j.stdout
#SBATCH --error={dump_dir}/logs/%j/%j.stderr

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

# Mimic the effect of "conda init", which doesn't work for scripts
eval "$({conda_exe} shell.bash hook)"
source activate {conda_env_path}

{go_to_code_dir}

export OMP_NUM_THREADS=1
export LAUNCH_WITH="SBATCH"
export DUMP_DIR={dump_dir}
srun {log_output} -n {tasks} -N {nodes_per_run} python -u -m {script} config=$DUMP_DIR/base_config.yaml
"""


def copy_dir(input_dir: str, output_dir: str) -> None:
    print(f"Copying : {input_dir}\n" f"to      : {output_dir} ...")
    assert os.path.isdir(input_dir), f"{input_dir} is not a directory"
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
    rsync_cmd = (
        f"rsync -arm --copy-links "
        f"--include '**/' "
        f"--include '*.py' "
        f"--exclude='*' "
        f"{input_dir}/ {output_dir}"
    )
    print(f"Copying command: {rsync_cmd}")
    subprocess.call([rsync_cmd], shell=True)
    print("Copy done.")


def retrieve_max_time_per_partition() -> Dict[str, int]:
    # retrieve partition max times (a bit slow)

    sinfo = json.loads(subprocess.check_output("sinfo --json", shell=True))["sinfo"]
    max_times: Dict[str, int] = {}

    for info in sinfo:
        if info["partition"]["maximums"]["time"]["infinite"]:
            max_times[info["partition"]["name"]] = 14 * 24 * 60  # 14 days
        else:
            max_times[info["partition"]["name"]] = info["partition"]["maximums"][
                "time"
            ][
                "number"
            ]  # in minutes

    return max_times


def validate_args(args) -> None:
    # Set maximum time limit if not specified
    if args.time == -1:
        max_times = retrieve_max_time_per_partition()
        args.time = max_times.get(
            args.partition, 3 * 24 * 60
        )  # Default to 3 days if not found
        print(
            f"No time limit specified, using max time for partitions: {args.time} minutes"
        )

    if args.constraint:
        args.constraint = f"#SBATCH --constraint={args.constraint}"

    if args.account:
        args.account = f"#SBATCH  --account={args.account}"

    if args.qos:
        args.qos = f"#SBATCH --qos={args.qos}"

    if getattr(args, "exclude", ""):
        args.exclude = f"#SBATCH --exclude={args.exclude}"

    if hasattr(args, "anaconda") and args.anaconda:
        if args.anaconda == "default":
            args.anaconda = (
                subprocess.check_output("which python", shell=True)
                .decode("ascii")
                .strip()
            )
        else:
            args.anaconda = f"{args.anaconda}/bin/python"
        assert os.path.isfile(args.anaconda)

    args.mem = args.mem or "0"

    assert args.partition
    assert args.ngpu > 0
    assert args.ncpu > 0
    assert args.nodes > 0
    assert args.time > 0
    assert args.partition


def launch_job(args: StoolArgs):
    # Set up args default and validate them depending on the cluster or partition requested
    validate_args(args)
    dump_dir = args.config["dump_dir"]
    job_name = args.config["name"]
    print("Creating directories...")
    os.makedirs(dump_dir, exist_ok=args.dirs_exists_ok or args.override)
    if args.override:
        confirm = input(
            f"Are you sure you want to delete the directory '{dump_dir}'? This action cannot be undone. (yes/no): "
        )
        if confirm.lower() == "yes":
            shutil.rmtree(dump_dir)
            print(f"Directory '{dump_dir}' has been deleted.")
        else:
            print("Operation cancelled.")
            return
    if args.copy_code:
        os.makedirs(f"{dump_dir}/code", exist_ok=args.dirs_exists_ok)
        print("Copying code ...")
        copy_dir(os.getcwd(), f"{dump_dir}/code")

    print("Saving config file ...")
    with open(f"{dump_dir}/base_config.yaml", "w") as cfg:
        cfg.write(OmegaConf.to_yaml(args.config))

    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = os.path.dirname(os.path.dirname(args.anaconda))
    log_output = (
        "-o $DUMP_DIR/logs/%j/%j_%t.out -e $DUMP_DIR/logs/%j/%j_%t.err"
        if not args.stdout
        else ""
    )
    sbatch = SBATCH_COMMAND.format(
        name=job_name,
        script=args.script,
        dump_dir=dump_dir,
        nodes=args.nodes,
        tasks=args.nodes * args.ngpu,
        nodes_per_run=args.nodes,
        ngpus=args.ngpu,
        ncpu=args.ncpu,
        mem=args.mem,
        qos=args.qos,
        account=args.account,
        constraint=args.constraint,
        exclude=args.exclude,
        time=args.time,
        partition=args.partition,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        log_output=log_output,
        go_to_code_dir=f"cd {dump_dir}/code/" if args.copy_code else "",
    )

    print("Writing sbatch command ...")
    with open(f"{dump_dir}/submit.slurm", "w") as f:
        f.write(sbatch)

    print("Submitting job ...")
    os.system(f"{args.launcher} {dump_dir}/submit.slurm")

    print("Done.")


if __name__ == "__main__":
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMTransformerArgs

    @dataclass
    class LMTransformerArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgs
    or just name=tictac for top level attributes.
    """
    args = OmegaConf.from_cli()
    args.config = OmegaConf.load(args.config)
    args = dataclass_from_dict(StoolArgs, args)
    launch_job(args)
