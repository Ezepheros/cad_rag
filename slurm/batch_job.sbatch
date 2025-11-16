#!/bin/bash
#SBATCH --job-name=EmbedOncaQwen8B       # Job name
#SBATCH --output=slurm_logs/embed_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=slurm_logs/embed_%j.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for Python)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs (if needed)
#SBATCH --mem=100G                      # Memory per node
#SBATCH --time=24:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=nlpgpo                 # Partition/queue name

# Load modules or activate environment
module load anaconda
conda activate legal_rag

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Move to your project directory
cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/cad_rag

# Run your Python training script
python embed_dataset.py
