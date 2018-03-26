#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=kp_base_tu
#SBATCH --output=kp_base_tu.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../model/train.py -cmode tuzhaopeng:wd_attn:kp_attn -ngpus 1 -bsize 4 -lr 0.0001 --output_folder kp_base_tu2 --max_kword_len 50 --max_abstr_len 1200 --max_cnt_kword 10 -env crc -warm /zfs1/hdaqing/saz31/keyphrase/kp_base_tu/log/model.ckpt-99999999

