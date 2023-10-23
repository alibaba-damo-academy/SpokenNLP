#!/bin/bash
source ~/.bashrc
conda activate ditto
run_shell() {
  NAME=$(date +%Y%m%d%H)_${short_name}_layer${layer}_head${head}_ditto
  cuda_id=0
  mkdir -p results/$NAME
  CUDA_VISIBLE_DEVICES=${cuda_id} python evaluation_ditto.py \
    --model_name_or_path ${model_name} \
    --pooler att_first_last \
    --task_set sts \
    --layer $layer \
    --head $head \
    --mode test 2>&1 | tee results/$NAME/log_test.txt
}

model_name=bert-base-uncased
#model_name=roberta-base
#model_name=google/electra-base-discriminator
#model_name=sentence-transformers/bert-base-nli-stsb-mean-tokens
if [ $model_name = "bert-base-uncased" ]; then
  layer=0
  head=9
  short_name="bert"
elif [ $model_name = "roberta-base" ]; then
  layer=0
  head=4
  short_name="roberta"
elif [ $model_name = "google/electra-base-discriminator" ]; then
  layer=0
  head=10
  short_name="electra"
elif [ $model_name = "sentence-transformers/bert-base-nli-stsb-mean-tokens" ]; then
  layer=2
  head=6
  short_name="sbert"
fi
run_shell

conda deactivate
