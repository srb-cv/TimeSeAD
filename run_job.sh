#!/bin/bash
#SBATCH --job-name=dmc-dataset
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --cpus-per-task=4        # CPU cores
#SBATCH --mem=32G                # RAM
#SBATCH --time=24:00:00          # max runtime

MODE=$1

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

