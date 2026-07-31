#!/bin/bash
#SBATCH --job-name=parley_RQ1
#SBATCH --partition=std
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd "$HOME/parley-uncertainty-modeling/PARLEY-2.0"

source venv/bin/activate

srun python RQ1_2.py