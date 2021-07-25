#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}
#HIDDEN_SIZE = ${14}

if [ $MODE == "train" ]
then

echo "Start Training......"

# bash run.sh train RotatE FB15k    0       0      1024        256               1000         24.0    1.0   0.0001 150000         16               -de
#               1     2      3       4      5        6          7                   8          9       10     11     12           13
#              mode model  dataset  GPU  saveid    batchsize   neg_sample_size  hidden_dim    gamma   alpha   lr    Max_steps  test_batchsize

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run_v2.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    ${14} ${15} ${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run_v2.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run_v2.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi