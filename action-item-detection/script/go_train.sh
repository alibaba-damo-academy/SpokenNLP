#!/bin/bash
PRETRAIN_DIR="../pretrain/structbert_base_english"
PRETRAIN_CKPT="model.ckpt-80000"
CLASS_INPUT="cls"   # cls, sep, token_avg, token_max
CLASS_MODEL="linear"      # linear
CONTEXT_TYPE="sentence+left+right"    # sentence, sentence+left, sentence+right, left+sentence+right
DEV_CONTEXT_TYPE="sentence+left+right"    # sentence, sentence+left, sentence+right, left+sentence+right
TEST_CONTEXT_TYPE="sentence+left+right"    # sentence, sentence+left, sentence+right, left+sentence+right
CONTEXT_WIDTH=1           # context width
LOSS_SMOOTH=False         # True, False
LOSS_TYPE="focal_loss"    # loss, focal_loss
THRESHOLD=0.5             # threshold
MAX_LENGTH=128            # max sequence length
DROP_TYPE="r-drop"  # r-drop, context-drop-fix, context-drop-dynamic
DROPOUT_RATE=0.3          # dropout rate
KL_ALPHA=4.0              # kl loss alpha
NOISY_TYPE="remain"          # skip/remain/update negative focus sentence with positive context

DATA_NAME="AMI"
DATA_DIR="../data/${DATA_NAME}/dataset"
TRAIN_FILE="train.txt"
DEV_FILE="dev.txt"
TEST_FILE="test.txt"

OUTPUT_DIR=$1
BEST_DIR=$2
OUTPUT_DIR="${OUTPUT_DIR:=output}"
BEST_DIR="${BEST_DIR:=best_model}"

BATCH_SIZE=$3
BATCH_SIZE=${BATCH_SIZE:=32}
LEARNING_RATE=$4
LEARNING_RATE=${LEARNING_RATE:=2e-5}
NUM_EPOCH=$5
NUM_EPOCH=${NUM_EPOCH:=2.0}

CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --task_name=meet  \
  --vocab_file=${PRETRAIN_DIR}/vocab.txt \
  --bert_config_file=${PRETRAIN_DIR}/bert_config.json \
  --init_checkpoint=${PRETRAIN_DIR}/${PRETRAIN_CKPT} \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --do_lower_case=True \
  --classifier_input=${CLASS_INPUT} \
  --classifier_model=${CLASS_MODEL} \
  --context_type=${CONTEXT_TYPE} \
  --dev_context_type=${DEV_CONTEXT_TYPE} \
  --test_context_type=${TEST_CONTEXT_TYPE} \
  --context_width=${CONTEXT_WIDTH} \
  --noisy_type=${NOISY_TYPE} \
  --threshold=${THRESHOLD} \
  --do_label_smoothing=${LOSS_SMOOTH} \
  --loss_type=${LOSS_TYPE} \
  --dropout_rate=${DROPOUT_RATE} \
  --kl_alpha=${KL_ALPHA} \
  --drop_type=${DROP_TYPE} \
  --data_dir=${DATA_DIR} \
  --record_dir=record/${DATA_NAME} \
  --train_file=${DATA_DIR}/${TRAIN_FILE} \
  --dev_file=${DATA_DIR}/${DEV_FILE} \
  --test_type="file" \
  --test_path=${DATA_DIR}/${TEST_FILE} \
  --predict_dir=${OUTPUT_DIR}/predict/file_test/ \
  --model_log_file=model.log.txt \
  --output_dir=${OUTPUT_DIR}/ \
  --best_model_dir=${BEST_DIR}/ \
  --do_export=False \
  --do_frozen=False \
  --export_dir=export/ \
  --max_seq_length=${MAX_LENGTH} \
  --train_batch_size=${BATCH_SIZE} \
  --eval_batch_size=32 \
  --learning_rate=${LEARNING_RATE} \
  --warmup_proportion=0.1 \
  --save_checkpoints_steps=100 \
  --train_summary_steps=10 \
  --num_train=${NUM_EPOCH} \
  --num_train_type="epoch" \

