#!/bin/bash

#SBATCH -A uppmax2026-1-123
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -J qwen_baseline
#SBATCH -o results/bm25.out
#SBATCH -e results/bm25.err

cd /gorilla/home/yuli6752/medical

source /home/yuli6752/miniconda3/etc/profile.d/conda.sh

conda activate medrag

export PYTHONNOUSERSITE=1

echo "Python path:"
which python

echo "Torch version:"
python -c "import torch; print(torch.__version__)"

python src/bm25_rag.py
