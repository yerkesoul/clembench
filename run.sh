#!/bin/bash

if [ ! $# -eq 2 ]; then
  echo "Please provide exactly two arguments: run.sh <game_name> <model_pair>"
  exit 1
fi

arg_game="$1"
arg_model="$2"

# Load and prepare path
# source venv/bin/activate
source prepare_path.sh

# Set temperature to 0.0
{ time python3 scripts/cli.py -m "$arg_model" -t 0.0 run "$arg_game"; } 2>&1 | tee runtime."$arg_game"."$arg_model".log
