#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
DEVICE=0
FIXMATCH_FOLDER="$HOME/project/SSLRS/"
SAVE_LOCATION="/scratch/fixmatch_results/" #Where tensorboard output will be written
SAVE_NAME="weigth_decay"                              #Options are wideResNet,efficientNet
DATASET=eurosat_rgb                          #Dataset to use
NUM_LABELS=250
UNLABELED_RATIO=7
BATCH_SIZE=32
N_EPOCH=500                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER * BATCH_SIZE / 32
NUM_EVAL_ITER=1000            #Number of iterations 
NUM_TRAIN_ITER=$(($N_EPOCH * $NUM_EVAL_ITER * BATCH_SIZE/ 32))
SEED=0
PRETRAINED=0
LR=0.03

#create save location
mkdir -p $SAVE_LOCATION

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER

NET=efficientnet-b2 
for WEIGHT_DECAY in 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001; do
	echo python train.py --weight_decay $WEIGHT_DECAY --rank 0 --gpu $DEVICE --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --num_classes -1 --net $NET --seed $SEED --uratio $UNLABELED_RATIO
	wait
done

WEIGHT_DECAY=0.0001

for NET in efficientnet-b3 efficientnet-b5; do
	echo python train.py --weight_decay $WEIGHT_DECAY --rank 0 --gpu $DEVICE --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --num_classes -1 --net $NET --seed $SEED --uratio $UNLABELED_RATIO
	wait
done

