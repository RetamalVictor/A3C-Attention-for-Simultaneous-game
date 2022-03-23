#!/bin/bash
#SBATCH --job-name="hello_world"
#SBATCH --partition=small (the default is debug)
#SBATCH --account=adminGrp (Uses the primary group of the user submitting the job)
#SBATCH --nodes=1 (can use min max notation )
#SBATCH --tasks-per-node=16
#SBATCH --time=00:00:05
#SBATCH --mem=1GB (There's no default, its REQUIRED)
#SBATCH --mail-user=myemail@gmail.com
#SBATCH --mail-type=BEGIN, END, FAIL, TIMELIMIT, REQUEUE, ALL, NONE
#SBATCH --workdir="/path/to/workdir" (defaults pwd)
#SBATCH --output=hello_world-%j.o (where %j is the job id; default is slurm-%j)
#SBATCH --error=hello_world-%j.e
#SBATCH --constraint="tengig" (constraint is )