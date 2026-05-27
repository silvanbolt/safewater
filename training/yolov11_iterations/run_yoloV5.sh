#!/bin/bash
#SBATCH --job-name=yolo_multi5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=17:00:00
#SBATCH --account=dslab_jobs

# Output and error files
#SBATCH --output=/work/courses/dslab/team7/multi-national/logs/yolo_train5_%j.out
#SBATCH --error=/work/courses/dslab/team7/multi-national/logs/yolo_train5_%j.err

# Load the module system
. /etc/profile.d/modules.sh

# Load CUDA and GCC
module add cuda/13.0
module add gcc/11

# Activate your Conda env
source /home/boltsi/miniconda3/bin/activate
conda activate conda_env

# Change to your project directory
cd /work/courses/dslab/team7/multi-national

# Run your Python script
python yoloV5.py
