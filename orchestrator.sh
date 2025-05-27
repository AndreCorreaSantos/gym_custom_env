#!/bin/bash

# Run training script
python train_grid_trail.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting evaluation..."
    python evaluate_grid_trail.py

    # Check if evaluation succeeded
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully. Running graphs notebook..."
        papermill graphs.ipynb graphs_output.ipynb
    else
        echo "Evaluation failed. Notebook will not run."
        exit 1
    fi
else
    echo "Training failed. Evaluation will not run."
    exit 1
fi
