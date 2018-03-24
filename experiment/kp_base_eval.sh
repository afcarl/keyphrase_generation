#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=kp_base_eval
#SBATCH --output=kp_base_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=10g

# Load modules
module restore

# Run the job
srun python ../model/eval.py -ngpus 1 --eval_mode truncate2000 -bsize 25 -beam 4 -lr 0.0001 --output_folder kp_base --max_kword_len 50 --max_abstr_len 1000 --max_cnt_kword 6 -env crc

