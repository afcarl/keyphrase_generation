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
srun python ../model/eval.py -cmode tuzhaopeng:wd_attn:kp_attn -ngpus 1 --eval_mode truncate2000 -bsize 35 -beam 1 -lr 0.0001 --output_folder kp_base_tu2 --max_kword_len 50 --max_abstr_len 1200 --max_cnt_kword 100 -env crc
