#!/bin/sh
#SBATCH -o /home/pbdang/Contest/SHREC22/OpenSet/SHOT/slurm_out/%j.out
python image_source.py --trte val --da oda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 50 --s 0