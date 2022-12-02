#!/usr/bin/env bash

source ~/.bashrc
conda activate modelscope

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

#for seed in 42 88 100 199 666
for seed in 88
do

    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)
    TASK=doc_key_sentence_extraction
    OUTPUT_DIR=./output/$TASK/ponet-docExtracitveSum-seed${seed}

    mkdir -p $OUTPUT_DIR
    LOGFILE=$OUTPUT_DIR/run.log

    # Allow multiple threads
    export OMP_NUM_THREADS=8

    MODEL_NAME_OR_PATH=damo/nlp_ponet_fill-mask_chinese-base

    echo $CUDA_VISIBLE_DEVICES >> ${LOGFILE}

    #sleep 0.5h

    # Use distributed data parallel
    # If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
#    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/extractive_summarization/ponet_extractive_summarization.py \
    python ./src/extractive_summarization/ponet_extractive_summarization.py \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --dataset_name ./datasets/AMC \
      --dataset_config_name $TASK \
      --metric_name ./metrics/extractive_summarization_eval \
      --save_total_limit 2 \
      --do_train True \
      --seed ${seed} \
      --do_eval True \
      --do_predict \
      --evaluation_strategy steps \
      --logging_steps 200 \
      --eval_steps 200 \
      --save_steps 200 \
      --load_best_model_at_end \
      --metric_for_best_model multi-ref-max_rouge-l_f \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 8 \
      --overwrite_output_dir \
      --learning_rate 5e-5 \
      --num_train_epochs 10 \
      --max_seq_length 4096 \
      --preprocessing_num_workers 5 \
      --return_entity_level_metrics True \
      --output_dir $OUTPUT_DIR >> ${LOGFILE} 2>&1
done