#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

FIXMATCH_FOLDER="$HOME/project/SSLRS/"
SAVE_LOCATION="/scratch/fixmatch_results" #Where tensorboard output will be written
SAVE_NAME="hyperparameters"
NET=efficientNet                              #Options are wideResNet,efficientNet
DATASET=eurosat_rgb                          #Dataset to use
NUM_LABELS=250
UNLABELED_RATIO=7
BATCH_SIZE=24
N_EPOCH=200                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER
NUM_EVAL_ITER=1000            #Number of iterations 
NUM_TRAIN_ITER=$(($N_EPOCH * $NUM_EVAL_ITER))
SEED=0

#create save location
mkdir -p $SAVE_LOCATION

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER

for LR in 0.03 0.1 0.2; do
    for WEIGHT_DECAY in 0.0001 0.0005 0.001; do
        echo python train.py --weight_decay $WEIGHT_DECAY --world-size 1 --rank 0 --multiprocessing-distributed --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --num_classes -1 --net $NET --seed $SEED --uratio $UNLABELED_RATIO
		wait
    done
done
