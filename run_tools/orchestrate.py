"""Orchestrate a case with slurm jobs, or run it locally without slurm.

This will set up all the necessary slurm jobs that will complete the run.

Usage: orchestrate.py PATH CASE TIME [--parts=PARTS | --bundle=BUNDLE] [--runexisting] [--mem=MEM]
                      [--memcollect=MEMCOLLECT] [--mailtype=MAILTYPE] [--nocollect]
       orchestrate.py --local PATH CASE [--runexisting]
       orchestrate.py (-h | --help)

Arguments:
    PATH    path to the scenario directory, which must contain case_definition.py
    CASE    case number
    TIME    specify maximum run time in DAYS-HH:MM:SS format

Options:
    -h --help                   show this help message
    --parts=PARTS               optionally, specify just some parts using sbatch --array syntax. By default all are run.
    --bundle=BUNDLE             how many parts to bundle in one slurm job,
                                the number of all parts must be divisible by BUNDLE. [default: 1]
    --runexisting               Set this flag to also run parts that already have results.
                                If this is not specified and neither --bundle nor --parts are used,
                                it avoids creating jobs for these parts in the first place.
    --mem=MEM                   memory in MB per part run [default: 2048].
    --memcollect=MEMCOLLECT     memory in MB for result collection step [default: 1024].
    --mailtype=MAILTYPE         mail-type option for sbatch. [default: FAIL,TIME_LIMIT].
    --nocollect                 Set this flag to skip the collection step. Useful if many orchestrates
                                are launched at the same time and collection is handled via a supercase.
    --local                     Run locally instead of using slurm.
"""
import os
import sys
import subprocess
from docopt import docopt
import re
import json

with open("run_config.json", "r") as f:
    run_config = json.load(f)


def assert_dir(path):
    """Check if `path` exists, and create it if it doesn't.

    Parameters
    ----------
    path : str
        The path to be checked/created.

    Returns
    -------
    None

    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    args = docopt(__doc__)
    path = args["PATH"]
    sys.path.append(os.path.abspath(path))
    import case_definition as cd

    case = int(args["CASE"])
    case_name = cd.name(case)
    subcase_name = cd.subcase_name(case)
    job_name = subcase_name + "_" + case_name
    result_path = os.path.join("results", cd.scenario_name, case_name, subcase_name)

    try:
        python_run_string = run_config["python_run_command"]
    except KeyError:
        python_run_string = "pipenv run python"

    python_run_command = python_run_string.split(" ")

    if args["--local"]:
        assert_dir(result_path)
        assert_dir(os.path.join(result_path, "parts"))
        nparts = cd.num_parts(case)
        base_command = python_run_command + ["run_tools/run_case.py"]
        if args["--runexisting"]:
            base_command.append("--runexisting")
        for part in range(nparts):
            run_command = base_command + [str(path), str(case), str(part)]
            submit = subprocess.run(run_command, capture_output=True)
            err = submit.stderr.decode("ascii")
            if err:
                print(err)
        collect_command = base_command + [
            str(path),
            str(case),
            "--collect",
        ]
        submit = subprocess.run(collect_command, capture_output=True)
        err = submit.stderr.decode("ascii")
        if err:
            print(err)
        sys.exit(0)

    email = run_config["notification_email"]
    parts = args["--parts"]
    bundle = int(args["--bundle"])
    if parts is None:
        if bundle == 1 and not args["--runexisting"]:
            parts_to_run = []
            nparts = cd.num_parts(case)
            for part_index in range(nparts):
                if not os.path.exists(
                    os.path.join(result_path, "parts", f"part{part_index}.csv")
                ):
                    parts_to_run += [str(part_index)]
            if not parts_to_run:
                print(
                    f"No parts for case {case} found that need to be run. Use --runexisting to run anyway."
                )
                sys.exit(0)
            if len(parts_to_run) == nparts:
                array_entry = f"0-{nparts - 1}"
            else:
                array_entry = ",".join(parts_to_run)
        else:
            nparts = cd.num_parts(case)
            if nparts % bundle != 0:
                raise ValueError(
                    f"The number of parts {nparts} must be divisible by --bundle={bundle}"
                )
            num_array_jobs = nparts // bundle
            array_entry = f"0-{num_array_jobs - 1}"
    else:
        array_entry = parts
    environment_setup_string = run_config["environment_setup"]
    if args["--runexisting"]:
        run_string = (
            f"{python_run_string} run_tools/run_case.py --runexisting {path} {case}"
        )
    else:
        run_string = f"{python_run_string} run_tools/run_case.py {path} {case}"
    if bundle == 1:
        run_instructions = f"{run_string} $SLURM_ARRAY_TASK_ID"
    else:
        run_instructions = "\n".join(
            [
                f"{run_string} $(({bundle} * $SLURM_ARRAY_TASK_ID + {i}))"
                for i in range(bundle)
            ]
        )

    sbatch_text = f"""#!/bin/bash

#SBATCH --job-name={job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time={args["TIME"]}              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args["--mem"]}              # Memory per cpu in MB (see also --mem)
#SBATCH --array={array_entry}
#SBATCH --output=out_files/%x_%a.out           # File to which standard out will be written
#SBATCH --error=out_files/%x_%a.err            # File to which standard err will be written
#SBATCH --mail-type={args["--mailtype"]}                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

{environment_setup_string}
{run_instructions}
"""
    assert_dir("out_files")
    assert_dir(result_path)
    assert_dir(os.path.join(result_path, "parts"))
    sbatch_file = os.path.join(result_path, f"run_case_{case}.sh")
    with open(sbatch_file, "w") as f:
        f.write(sbatch_text)
    submit1 = subprocess.run(["sbatch", sbatch_file], capture_output=True)

    out1 = submit1.stdout.decode("ascii")
    err1 = submit1.stderr.decode("ascii")
    if err1:
        raise RuntimeError(err1)
    if not args["--nocollect"]:
        jid1 = re.search("([0-9]+)", out1).group(1)
        collect_text = f"""#!/bin/bash

#SBATCH --job-name=c_{job_name}     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-00:05:00            # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu={args["--memcollect"]}              # Memory per cpu in MB (see also --mem)
#SBATCH --output=out_files/%x.out           # File to which standard out will be written
#SBATCH --error=out_files/%x.err            # File to which standard err will be written
#SBATCH --mail-type={args["--mailtype"]}                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={email}   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

{environment_setup_string}
{python_run_string} run_tools/run_case.py --collect {path} {case}
"""
        collect_file = os.path.join(result_path, f"collect_case_{case}.sh")
        with open(collect_file, "w") as f:
            f.write(collect_text)
        submit2 = subprocess.run(
            [
                "sbatch",
                f"--dependency=afterany:{jid1}",
                "--deadline=now+14days",
                collect_file,
            ],
            capture_output=True,
        )
        out2 = submit2.stdout.decode("ascii")
        err2 = submit2.stderr.decode("ascii")
        if err2:
            raise RuntimeError(err2)
