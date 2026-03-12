#!/bin/bash
# Double-click this file to launch maskAnnotate

# Change to the script's directory
cd "$(dirname "$0")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fly2p

# Launch the GUI
python run_gui.py
