#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=kp_base
#SBATCH --output=kp_base.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=10g

# Load modules
module restore

# Run the job
srun python ../model/train.py -ngpus 1 -bsize 5 -lr 0.0001 --output_folder kp_base --max_kword_len 30 --max_abstr_len 1000 --max_cnt_kword 6 -env crc

