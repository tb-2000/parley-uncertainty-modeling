#!/bin/bash
#SBATCH --job-name=parley_RQ1
#SBATCH --partition=longrun
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
<<<<<<< HEAD
#SBATCH --mem=360G
=======
#SBATCH --mem=130G
>>>>>>> 9f236002892fec5e3bc6f93b7d6f30ef9eaf93c6
#SBATCH --time=4-00:00:00

set -euo pipefail

cd "$HOME/parley-uncertainty-modeling/PARLEY-2.0"

source venv/bin/activate

srun python RQ1_2.py