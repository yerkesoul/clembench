#!/bin/bash
# Usage: ./pipeline_huggingfaces.sh
# Preparation: ./setup.sh && ./setup_hf.sh
echo
echo "==================================================="
echo " PIPELINE: Starting $0"
echo "==================================================="
echo
# the key file has to exist
cp key_default.json key.json
# only self-play and single player for now
models=(
  "koala-13b"
  "vicuna-13b"
  "falcon-40b"
  "oasst-12b"
)
source venv_hf/bin/activate
source prepare_path.sh
# Run the benchmark (one model after the other; they consume limited GPU memory)
# but play all games directly (since the loading into memory takes so much time)
for model in "${models[@]}"; do
  { time python3 scripts/cli.py -m "$model" -t 0.0 run all; } 2>&1 | tee runtime.all."$model".log
done
echo "========================================================================="
echo "PIPELINE: Finished"
echo "========================================================================="
