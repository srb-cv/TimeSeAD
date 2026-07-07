#!/bin/bash
#SBATCH -t 1-00:00:00                    # time limit format "days-hours:minutes:seconds"
#SBATCH --mem=30G                        # reserve 64GB of memory
#SBATCH -J cheetah_ad                   # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH --ntasks=1                       # total number of tasks across all nodes
#SBATCH --cpus-per-task=3                # use cpus-per-task number threads per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --output=sbatch_logs/week_20/sbatch_%j_out.log         # capture output
#SBATCH --error=sbatch_logs/week_20/sbatch_%j_err.log         # and error streams
#SBATCH --gres=gpu:v100:1
#SBATCH --account=RPTU-ML-VAD    # run with high priority using VAD account
#SBATCH --partition=dgx

set -euo pipefail

MODE=${1:-}
if [ -z "$MODE" ]; then
    echo "Invalid mode. Use: train, grid, or evaluate"
    exit 1
fi
shift

export PYTHONUNBUFFERED=1

if [ -z "${LOGDIR:-}" ]; then
    JOB_ID=${SLURM_JOB_ID:-local}
    TIMESTAMP=$(date "+%Y-%m-%d_%Hh%Mm")
    LOGDIR="outputs/slurm/${MODE}/${TIMESTAMP}_job-${JOB_ID}"
    echo "WARNING: LOGDIR not set; using default LOGDIR=$LOGDIR"
fi

if [ "$MODE" = "train" ]; then
    echo "Running training..."
    ENTRYPOINT="experiments_hydra/supervised/train_dsad.py"
    SOURCE_CONFIG="experiments_hydra/configs/dmc/supervised/train_dsad.yaml"
elif [ "$MODE" = "grid" ]; then
    echo "Running grid search..."
    ENTRYPOINT="experiments_hydra/grid_search.py"
    SOURCE_CONFIG="experiments_hydra/configs/grid_search/default.yaml"
elif [ "$MODE" = "evaluate" ]; then
    echo "Running evaluating"
    ENTRYPOINT="experiments_hydra/evaluate.py"
    SOURCE_CONFIG="experiments_hydra/configs/evaluate/default.yaml"
else
    echo "Invalid mode. Use: train, grid, or evaluate"
    exit 1
fi

CONFIG_DIR="$LOGDIR/config"
mkdir -p "$CONFIG_DIR"
cp "$SOURCE_CONFIG" "$CONFIG_DIR/config.yaml"

echo "Logdir: $LOGDIR"
echo "Copied config: $SOURCE_CONFIG -> $CONFIG_DIR/config.yaml"

uv run python3 "$ENTRYPOINT" \
    --config-path "$PWD/$CONFIG_DIR" \
    --config-name config \
    hydra.run.dir="$LOGDIR" \
    "$@"
