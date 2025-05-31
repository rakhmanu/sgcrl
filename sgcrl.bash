#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=64g
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output=/work/rleap1/ulzhalgas.rakhman/bin%A.txt

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/ulzhalgas.rakhman/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python lp_contrastive.py
