#!/bin/bash

# bash go_train.sh output_1 best_model_1
# bash go_predict.sh file test best_model_1 0.5

NUM_MODEL=$1
NUM_MODEL="${NUM_MODEL:=5}"
NUM_SET=$2
NUM_SET=${NUM_SET:=4}

MODEL_ID=1
trap "exit" INT
while [ ${MODEL_ID} -le ${NUM_MODEL} ]; do
    SET_ID=1
    for BATCH_SIZE in 32; do
        for LEARNING_RATE in 2e-5 1e-5; do
            for NUM_EPOCH in 2 3; do
                echo "### Training Repeat Model ${MODEL_ID}/${NUM_MODEL} # Set ${SET_ID}/${NUM_SET} With ${BATCH_SIZE} ${LEARNING_RATE} ${NUM_EPOCH}"
                bash go_train.sh output${MODEL_ID}_set${SET_ID} best${MODEL_ID}_set${SET_ID} ${BATCH_SIZE} ${LEARNING_RATE} ${NUM_EPOCH} -e
                bash go_predict.sh file test best${MODEL_ID}_set${SET_ID} 0.5 -e
                bash go_predict.sh file dev best${MODEL_ID}_set${SET_ID} 0.5 -e
                ((SET_ID++))
            done
        done
    done
    ((MODEL_ID++))
done

python average_performance.py ${NUM_MODEL} ${NUM_SET}




