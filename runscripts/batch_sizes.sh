#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

FIXMATCH_FOLDER="$HOME/project/SSLRS/"
SAVE_LOCATION="/scratch/fixmatch_results" #Where tensorboard output will be written
NET=efficientNet                              #Options are wideResNet,efficientNet
SAVE_NAME="batch_size"
DATASET=eurosat_rgb                          #Dataset to use
SEED=0
NUM_LABELS=250
UNLABELED_RATIO=7
WEIGHT_DECAY=0.0005
N_EPOCH=200                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER
NUM_EVAL_ITER=1000            #Number of iterations 
NUM_TRAIN_ITER=$(($N_EPOCH * $NUM_EVAL_ITER))

#create save location
mkdir -p $SAVE_LOCATION

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER

for BATCH_SIZE in 4 8 12 16 24 32; do
    echo python train.py --world-size 1 --rank 0 --multiprocessing-distributed --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --save_dir $SAVE_LOCATION --batch_size $BATCH_SIZE --num_labels $NUM_LABELS --save_name $SAVE_NAME --dataset $DATASET --num_classes -1 --weight_decay $WEIGHT_DECAY --net $NET --seed $SEED --uratio $UNLABELED_RATIO --overwrite
    wait
done