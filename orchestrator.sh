#!/bin/bash

# Run training script
python train_grid_trail.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting evaluation..."
    python evaluate_grid_trail.py
else
    echo "Training failed. Evaluation will not run."
    exit 1
fi
