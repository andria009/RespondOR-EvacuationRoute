#!/bin/bash
#SBATCH --job-name=respondor_evacuation
#SBATCH --output=logs/respondor_%j.out
#SBATCH --error=logs/respondor_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=compute

# ============================================================
# RespondOR-EvacuationRoute HPC Job Script (SLURM + MPI)
#
# Usage:
#   sbatch hpc/slurm_job.sh [config] [--benchmark]
#
# Examples:
#   sbatch hpc/slurm_job.sh
#   sbatch hpc/slurm_job.sh configs/disaster_scenario.yaml
#   sbatch hpc/slurm_job.sh configs/disaster_scenario.yaml --benchmark
# ============================================================

# ---- Environment ----
module purge
module load python/3.11
module load openmpi/4.1        # or intel-mpi/2021 — adjust for your cluster

# Activate virtual environment
source "$HOME/.pyenv/versions/respondor-evroute/bin/activate"

# Set working directory to submission dir
cd "$SLURM_SUBMIT_DIR"

# Create log directory
mkdir -p logs

# ---- Parse arguments ----
CONFIG="${1:-configs/disaster_scenario.yaml}"
N_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
OUTPUT_DIR="output/hpc_run_${SLURM_JOB_ID}"

echo "============================================"
echo " RespondOR HPC Run (MPI)"
echo " Job ID:    $SLURM_JOB_ID"
echo " Nodes:     $SLURM_NNODES"
echo " MPI ranks: $N_TASKS  ($SLURM_NTASKS_PER_NODE per node)"
echo " Config:    $CONFIG"
echo " Output:    $OUTPUT_DIR"
echo "============================================"

# ---- Distributed route optimization via MPI ----
# srun launches one MPI process per task; rank 0 is master, others are workers.
# All ranks run src/main.py — rank 0 loads data and broadcasts; workers
# receive their village partition, compute routes, and exit after gathering.
srun --mpi=pmix \
     -n "$N_TASKS" \
     python -m src.main \
         --config "$CONFIG" \
         --mode hpc \
         --output-dir "$OUTPUT_DIR" \
         --log-level INFO

EXIT_CODE=$?

# ---- Optional benchmark comparison ----
if [ "${2:-}" == "--benchmark" ]; then
    echo "Running mode comparison benchmark..."
    python -m src.main \
        --config "$CONFIG" \
        --benchmark \
        --output-dir "${OUTPUT_DIR}/benchmark"
fi

echo "Job completed with exit code $EXIT_CODE"
exit $EXIT_CODE
