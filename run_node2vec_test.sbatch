#!/bin/bash
#
#SBATCH --job-name=safeplate-test
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=01:00:00
#
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# -----------------------------
# Environment setup
# -----------------------------
source /etc/profile.d/modules.sh

# -----------------------------
# Activate your virtual env
# -----------------------------
source venv/bin/activate

# -----------------------------
# Run the script (NO NODE2VEC YET)
# -----------------------------
python scripts/03_train_node2vec.py
