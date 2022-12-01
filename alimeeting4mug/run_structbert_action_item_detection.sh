#!/usr/bin/env bash

source ~/.bashrc
conda activate modelscope

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1
#for seed in 42 88 100 199 666
for seed in 42
do
    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)
    TASK=action_detection
    OUTPUT_DIR=./$TASK/structbert-actionItemDetection-seed${seed}

    mkdir -p $OUTPUT_DIR
    LOGFILE=$OUTPUT_DIR/`date +%Y%m%d%H`.log

    # Allow multiple threads
    export OMP_NUM_THREADS=8

    MODEL_NAME_OR_PATH=damo/nlp_structbert_backbone_base_std

    echo $CUDA_VISIBLE_DEVICES >> ${LOGFILE}

    #sleep 0.5h

    # Use distributed data parallel
    # If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
    #python ./topic_segment/topic_segmentation_sentence_labeling.py \
    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/action_item_detection/structbert_meeting_action.py \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --dataset_name ./datasets/AMC \
      --dataset_config_name $TASK \
      --metric_name ./metrics/classification/ \
      --save_total_limit 2 \
      --do_train True \
      --seed ${seed} \
      --do_eval True \
      --do_predict \
      --evaluation_strategy steps \
      --logging_steps 100 \
      --eval_steps 100 \
      --save_steps 100 \
      --load_best_model_at_end \
      --metric_for_best_model overall_f1 \
      --per_device_train_batch_size 32 \
      --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --preprocessing_num_workers 5 \
      --output_dir $OUTPUT_DIR >> ${LOGFILE} 2>&1
done
