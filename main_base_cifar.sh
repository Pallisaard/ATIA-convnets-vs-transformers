#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --gres=gpu:1
#SBATCH --job-name=atia-conv-cif10
#number of independent tasks we are going to start in this script
# noope array 1-10%3
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=6
#the ammount of memory allocated
#SBATCH --mem=12000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=8:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program
# python --version > echo
echo $CUDA_VISIBLE_DEVICES

echo "Running convnext training on $CUDA_VISIBLE_DEVICES"

python main.py --model "resnet50" --dataset "cifar10" --data_path "data/datasets/cifar10/" --job_id "${SLURM_ARRAY_TASK_ID}" --num_workers 4 --lr 0.00001 --epochs 1
