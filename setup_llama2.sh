#!/bin/bash
python3 -m venv venv_llama2
source venv_llama2/bin/activate
pip3 install -r requirements.txt
pip3 install -r requirements_llama2.txt

mkdir llama2

echo "Please run download.sh from https://github.com/facebookresearch/llama/tree/main in the llama2 directory to acquire the Llama-2 model weights and tokenizer."