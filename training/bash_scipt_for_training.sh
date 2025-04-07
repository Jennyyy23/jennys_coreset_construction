#!/bin/bash -l
#
# time
# ADAPT TIME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --time=24:00:00
#
# job name
# ADAPT JOB NAME FOR PARALLEL JOBS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --job-name=LVH_all
# send email when the job begins, ends, or fails
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com
#
# specify log output file
# Log file for stdout
# ADAPT NAME !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --output=/home/hpc/iwai/iwai110h/logs/LVH/all_checkpoint_2.log   
# Log file for stderr
#SBATCH --error=/home/hpc/iwai/iwai110h/logs/LVH/all_checkpoint_2_error.log
#
# allocate one GPU
#SBATCH --gres=gpu:1
#
# cluster
#SBATCH --clusters=tinygpu
#
# do not export environment variables
#SBATCH --export=NONE
#
# do not export environment variables
unset SLURM_EXPORT_ENV
#
# load python module
module load python/3.10-anaconda
module load cuda
#
# activate environment
conda activate /home/woody/iwai/iwai110h/software/private/conda/envs/jenny1 
#
# Ensure the latest command paths and disk writes
sync
hash -r
#
# ADAPT DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# copy data to TMPDIR
#
# IMAGENET
# validation set
# unzip "/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/zip_official/test_val_all_random/IN_val.zip" -d "$TMPDIR/validationset"
# # test set 
# unzip "/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/zip_official/test_val_all_random/IN_test.zip" -d "$TMPDIR/testset"
# # * * * TRAINING SETS * * *
# unzip "/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/zip_official/distillation_frames.zip" -d "$TMPDIR/trainingset"
#
# ECHONET PEDIATRIC
# # validation set
# unzip "/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/EP_val.zip" -d "$TMPDIR/validationset"
# # test set 
# unzip "/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/EP_test.zip" -d "$TMPDIR/testset"
# # * * * TRAINING SETS * * *
# unzip "/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/EP_random_DB2reduced.zip" -d "$TMPDIR/trainingset"
#
# ECHONET DYNAMIC
# validation set
# unzip "/home/woody/iwai/iwai110h/EchoNet-Dynamic/zip_files_official_split/zip_all_random_val_test/ED_val.zip" -d "$TMPDIR/validationset"
# # test set 
# unzip "/home/woody/iwai/iwai110h/EchoNet-Dynamic/zip_files_official_split/zip_all_random_val_test/ED_test.zip" -d "$TMPDIR/testset"
# # * * * TRAINING SETS * * *
# unzip "/home/woody/iwai/iwai110h/EchoNet-Dynamic/zip_files_official_split/random_reduced_10percent.zip" -d "$TMPDIR/trainingset"
# 
# EchoNet LVH
# validation set 
unzip '/home/woody/iwai/iwai110h/EchoNet-LVH/LVH_val.zip' -d "$TMPDIR/validationset"
# test set 
unzip '/home/woody/iwai/iwai110h/EchoNet-LVH/LVH_test.zip' -d "$TMPDIR/testset"
# training set 
unzip '/home/woody/iwai/iwai110h/EchoNet-LVH/LVH_all_frames.zip' -d "$TMPDIR/trainingset"

# Run the training script (it will resume from checkpoint if available)
timeout 23h python training/ResNet18_training_script.py

# If the script times out, resubmit the job
if [[ $? -eq 124 ]]; then
  sbatch bash_coreset.sh
fi