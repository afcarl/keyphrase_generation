#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=kp_base_tu_eval
#SBATCH --output=kp_base_tu_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=10g

# Load modules
module restore

# Run the job
srun python ../model/eval.py -cmode tuzhaopeng --eval_mode truncate2000 -ngpus 1 -bsize 50 -beam 1 -lr 0.0001 --output_folder kp_base_tu --max_kword_len 30 --max_abstr_len 1000 --max_cnt_kword 50 -env crc
