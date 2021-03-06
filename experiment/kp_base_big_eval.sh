#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=kp_base_big_eval
#SBATCH --output=kp_base_big_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../model/eval.py -ngpus 1 --eval_mode truncate2000 -bsize 5 -beam 4 -dim 512 -lr 0.0001 -nh 8 -nhl 6 -nel 6 -ndl 6 --output_folder kp_base_big --max_kword_len 50 --max_abstr_len 1000 --max_cnt_kword 6 -env crc &
