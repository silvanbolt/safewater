#!/bin/bash
#SBATCH --job-name=student_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --account=dslab_jobs

# Output and error files
#SBATCH --output=/work/courses/dslab/team7/teacher_student/logs/student_eval_%j.out
#SBATCH --error=/work/courses/dslab/team7/teacher_student/logs/student_eval_%j.err

# Load the module system
. /etc/profile.d/modules.sh

# Load CUDA and GCC
module add cuda/13.0
module add gcc/11

# Activate your Conda env
source /home/boltsi/miniconda3/bin/activate
conda activate conda_env

# Change to your project directory
cd /work/courses/dslab/team7/teacher_student

# Run your Python script
python eval_student_roc.py

