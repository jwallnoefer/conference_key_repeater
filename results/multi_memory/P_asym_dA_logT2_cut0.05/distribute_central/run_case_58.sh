#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=distribute_central_P_asym_dA_logT2_cut0.05     # Job name, will show up in squeue output
#SBATCH --ntasks=1                     # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=01-10:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=2048              # Memory per cpu in MB (see also --mem)
#SBATCH --array=0-399
#SBATCH --output=out_files/%x_%a.out           # File to which standard out will be written
#SBATCH --error=out_files/%x_%a.err            # File to which standard err will be written
#SBATCH --mail-type=FAIL,TIME_LIMIT                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=janka.memmen@tu-berlin.com   # Email to which notifications will be sent
#SBATCH --qos=standard

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

module add python/3.9.13
pipenv run python run_tools/run_case.py scenarios/multi_memory 58 $SLURM_ARRAY_TASK_ID
