#!/bin/bash
#SBATCH -t 1-00:00:00                    # time limit format "days-hours:minutes:seconds"
#SBATCH --mem=30G                        # reserve 64GB of memory
#SBATCH -J cheetah_ad                   # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH --ntasks=1                       # total number of tasks across all nodes
#SBATCH --cpus-per-task=3                # use cpus-per-task number threads per taks
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --output=sbatch_logs/week_17/sbatch_%j_out.log         # capture output
#SBATCH --error=sbatch_logs/week_17/sbatch_%j_err.log         # and error streams
#SBATCH --gres=gpu:v100:1
#SBATCH --account=RPTU-ML-VAD    # run with high priority using VAD account

MODE=$1
export PYTHONUNBUFFERED=1

if [ "$MODE" = "train" ]; then
    echo "Running training..."
    uv run python3 experiments_hydra/prediction/train_lstm_prediction_filonov.py
elif [ "$MODE" = "grid" ]; then
    echo "Running grid search..."
    uv run python3 experiments_hydra/grid_search.py
elif [ "$MODE" = "evaluate" ]; then
    echo "Running evaluating"
    uv run python3 experiments_hydra/evaluate.py
else
    echo "Invalid mode. Use: train or grid"
    exit 1
fi

# Run training