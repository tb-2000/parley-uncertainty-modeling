#!/bin/bash
#SBATCH --job-name=parleyGaussian
#SBATCH --partition=compute
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

set -euo pipefail

cd "$HOME/parley-uncertainty-modeling/PARLEY-2.0"

source venv/bin/activate

srun python RQ3_gaussian_scaling.py