#!/bin/bash
PRETRAIN_DIR="../pretrain/structbert_base_english"
PRETRAIN_CKPT="model.ckpt-80000"
CLASS_INPUT="cls"    # cls, sep, token_avg, token_max
CLASS_MODEL="linear"       # linear
CONTEXT_TYPE="sentence+left+right"    # sentence, sentence+left, sentence+right, left+sentence+right
TEST_CONTEXT_TYPE="sentence+left+right"    # sentence, sentence+left, sentence+right, left+sentence+right
CONTEXT_WIDTH=1           # context width
THRESHOLD=$4              # threshold
MAX_LENGTH=128            # max sequence length

DATA_NAME="AMI"
DATA_DIR="../data/${DATA_NAME}/dataset"
TEST_TYPE=$1    # file
TEST_NAME=$2    # test
MODEL_DIR=$3    # best model dir; default best_model

TEST_TYPE="${TEST_TYPE:=file}"
TEST_NAME="${TEST_NAME:=test}"
MODEL_DIR="${MODEL_DIR:=best_model}"
THRESHOLD="${THRESHOLD:=0.5}"

if [ ${TEST_TYPE} = "dir" ];then
    TEST_PATH=${DATA_DIR}/${TEST_NAME}/
    RECORD_PATH=record/${DATA_NAME}/${TEST_NAME}/
elif [ ${TEST_TYPE} = "file" ];then
    TEST_PATH=${DATA_DIR}/${TEST_NAME}.txt
    RECORD_PATH=record/${DATA_NAME}/
else
    TEST_PATH=${DATA_DIR}/${TEST_NAME}
    RECORD_PATH=record/${DATA_NAME}/
fi

CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --task_name=meet  \
    --vocab_file=${PRETRAIN_DIR}/vocab.txt \
    --bert_config_file=${PRETRAIN_DIR}/bert_config.json \
    --init_checkpoint=${PRETRAIN_DIR}/${PRETRAIN_CKPT} \
    --do_predict=True \
    --do_lower_case=True \
    --classifier_input=${CLASS_INPUT} \
    --classifier_model=${CLASS_MODEL} \
    --context_type=${CONTEXT_TYPE} \
    --test_context_type=${TEST_CONTEXT_TYPE} \
    --context_width=${CONTEXT_WIDTH} \
    --threshold=${THRESHOLD} \
    --data_dir=${DATA_DIR} \
    --test_type=${TEST_TYPE} \
    --test_path=${TEST_PATH} \
    --record_dir=${RECORD_PATH} \
    --predict_dir=${MODEL_DIR}/predict/${TEST_TYPE}_${TEST_NAME}/ \
    --model_log_file=model.log.txt \
    --output_dir=${MODEL_DIR}/ \
    --best_model_dir=${MODEL_DIR}/ \
    --max_seq_length=${MAX_LENGTH} \
    --predict_batch_size=32 \
    --learning_rate=2e-5



