#!/bin/bash
#SBATCH --job-name=sac_RL
#SBATCH --cpus-per-task=4
#SBATCH --partition=week
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cyprian.bohojlo@student.uni-tuebingen.de  

# Define project and scratch directories
PROJECT_DIR="/scratch/$SLURM_JOB_ID/sac_rl"
SUBMIT_DIR="$(pwd)"

# Copy project files to scratch
mkdir -p $PROJECT_DIR
cp -r /home/stud435/hockey_laseRL/* $PROJECT_DIR

# Copy uploaded files to the project directory
cp /mnt/data/environment.py $PROJECT_DIR/
cp /mnt/data/networks.py $PROJECT_DIR/
cp /mnt/data/replay.py $PROJECT_DIR/
cp /mnt/data/sac.py $PROJECT_DIR/
cp /mnt/data/train.py $PROJECT_DIR/
cp /mnt/data/training_sac_agent.py $PROJECT_DIR/

# Change to the scratch directory
cd $PROJECT_DIR

# Execute within Singularity container
singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg bash -c "
    pip install gymnasium torch==2.4.0 numpy==1.26.4 matplotlib --quiet
    pip install git+https://github.com/martius-lab/hockey-env.git --quiet
    
    # Execute the training script
    python training_sac_agent.py --mode weak_opponent --max_episodes 20000 --max_steps 500 \
        --save_path ./models --checkpoint_mode none --device cuda
"

# Copy results back to submission directory
mkdir -p $SUBMIT_DIR/models
cp -r $PROJECT_DIR/models/* $SUBMIT_DIR/models/ 2>/dev/null || true

# Copy your new score_logs.txt back
cp $PROJECT_DIR/score_logs.txt $SUBMIT_DIR/ 2>/dev/null || true
