#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

FIXMATCH_FOLDER="$HOME/project/SSLRS/"
SAVE_LOCATION="/scratch/fixmatch_results" #Where tensorboard output will be written
SAVE_NAME="seeds_run"
NET=efficientNet                              #Options are wideResNet,efficientNet
DATASET=eurosat_rgb                          #Dataset to use
NUM_LABELS=250
UNLABELED_RATIO=7
BATCH_SIZE=24								 #Batch size to use
WEIGHT_DECAY=0.0005
N_EPOCH=32                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER
NUM_EVAL_ITER=1000            #Number of iterations 
NUM_TRAIN_ITER=$(($N_EPOCH * $NUM_EVAL_ITER))

#create save location
mkdir -p $SAVE_LOCATION

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER

for SEED in 0 1 2 3 4 5; do
   echo python train.py --world-size 1 --rank 0 --multiprocessing-distributed --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --save_dir $SAVE_LOCATION --save_name $SAVE_NAME --weight_decay $WEIGHT_DECAY --batch_size $BATCH_SIZE --num_labels $NUM_LABELS --dataset $DATASET --num_classes -1 --net $NET --seed $SEED --uratio $UNLABELED_RATIO
   wait
done
