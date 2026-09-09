#!/bin/bash
#SBATCH --job-name=parleyGaussian
#SBATCH --partition=compute
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00

set -euo pipefail

cd "$HOME/parley-uncertainty-modeling/PARLEY-2.0"

source venv/bin/activate
srun python RQ1_2.py