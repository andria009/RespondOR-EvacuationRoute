#!/bin/bash
# ============================================================
# RespondOR-EvacuationRoute — Full Benchmark (SLURM)
#
# Runs benchmark_all.py across all scenarios × execution modes
# on 2 nodes × 128 cores using cached OSM + InaRISK data.
#
# Pre-requisite: run benchmark_extraction first to warm caches:
#   python -m experiments.benchmark_extraction --no-cache --skip-inarisk-edges
#
# Submit:
#   sbatch hpc/slurm_benchmark.sh
#
# Resume interrupted run:
#   sbatch hpc/slurm_benchmark.sh --resume
#
# Dry-run (print commands only):
#   sbatch hpc/slurm_benchmark.sh --dry-run
# ============================================================

#SBATCH --job-name=respondor_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# 2 nodes, 4 MPI tasks total (2 per node), 64 CPUs per task
# → each node: 2 tasks × 64 CPUs = 128 cores available
# → total: 4 tasks × 64 CPUs = 256 cores
#
# Parallel modes  use up to 128 workers on node 0
# HPC modes       use up to 4 MPI ranks × 64 workers = 256 cores across 2 nodes
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --exclusive                    # exclusive access for consistent timing

#SBATCH --mem=0                        # use all available memory on each node
#SBATCH --time=24:00:00                # wall limit — large scenarios (Merapi) take several hours
#SBATCH --partition=compute            # adjust to your cluster partition name

# ---- Environment setup ----
module purge
module load python/3.11                # adjust to your cluster's Python module
module load openmpi/4.1               # or intel-mpi/2021 — must support PMIx

# Activate Python environment (conda or pyenv)
# Option A — conda:
# source activate base
# Option B — pyenv virtualenv:
# source "$HOME/.pyenv/versions/respondor-evroute/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- Parse optional flags passed to sbatch ----
EXTRA_ARGS="${*}"        # e.g. --resume or --dry-run

# ---- Summary ----
echo "========================================================"
echo " RespondOR Benchmark"
echo " Job ID:        $SLURM_JOB_ID"
echo " Nodes:         $SLURM_NNODES  ($SLURM_NODELIST)"
echo " Tasks:         $SLURM_NTASKS  ($SLURM_NTASKS_PER_NODE per node)"
echo " CPUs/task:     $SLURM_CPUS_PER_TASK"
echo " Total CPUs:    $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo " Results:       output/benchmark_results.json"
echo " Extra args:    ${EXTRA_ARGS:-none}"
echo "========================================================"

# ---- Execution matrix ----
#
# Parallel modes  — single-node multiprocessing (run on head node)
#   parallel_1w  through parallel_128w
#
# HPC modes       — MPI across 2 nodes, each rank with its own ProcessPoolExecutor
#   hpc_2r_*  : 1 MPI rank per node  (2 ranks total)
#   hpc_4r_*  : 2 MPI ranks per node (4 ranks total)
#   workers   : 8, 16, 32, 64, 128 per rank
#
# Total modes: 1 naive + 8 parallel + 10 HPC = 19 modes × 10 scenarios = 190 runs
#
python -m experiments.benchmark_all \
    --parallel-workers 1 2 4 8 16 32 64 128 \
    --hpc-ranks 2 4 \
    --hpc-workers 8 16 32 64 128 \
    --mpi-launcher srun \
    --timeout 7200 \
    --resume \
    $EXTRA_ARGS

EXIT_CODE=$?
echo "Benchmark finished with exit code $EXIT_CODE"
exit $EXIT_CODE
