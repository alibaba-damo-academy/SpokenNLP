export CUDA_VISIBLE_DEVICES=0,1
NUM_GPU=2
echo $CUDA_VISIBLE_DEVICES
echo $NUM_GPU

PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

metric_name=./src/metrics/seqeval.py
model_root_folder=./pretrained_models

# max_seq_length=512
max_seq_length=2048

# value should be folder name of pretrained_model
# model_name=bert_base
# model_name=electra_base
# model_name=bigbird_base
model_name=longformer_base

dataset=wiki_section
dataset=wiki_section_disease
# dataset=wiki727k

dataset_cache_dir=./cached_data/${dataset}_${model_name}_${max_seq_length}

# num_train_epochs=3
num_train_epochs=5
lr=5e-5
per_device_train_batch_size=2
gradient_accumulation_steps=2

do_da_ts=True
do_cssl=True
do_tssp=True

ts_loss_weight=1.0

# cl_loss_weight=0.0
cl_loss_weight=0.5
cl_temp=0.1
cl_anchor_level=eop_list
cl_positive_k=1
cl_negative_k=3

# tssp_loss_weight=0.0
tssp_loss_weight=1.0    # for wiki_section
# tssp_loss_weight=0.5    # for wiki727k

for seed in 42 59 88; do
  
  bs=`expr $NUM_GPU \* $per_device_train_batch_size \* $gradient_accumulation_steps`
  currentTime=`date "+%Y-%m-%d_%H:%M:%S"`
  OUTPUT_DIR=./output/${model_name}-finetune-${dataset}/seed${seed}-seq${max_seq_length}-lr${lr}-epoch${num_train_epochs}-bs${bs}-ts${ts_loss_weight}-tssp${tssp_loss_weight}-cl${cl_loss_weight}-${currentTime}
  mkdir -p $OUTPUT_DIR
  LOGFILE=$OUTPUT_DIR/run.log
  echo "write log into "${LOGFILE}
  echo $CUDA_VISIBLE_DEVICES >> ${LOGFILE}

  # python ./src/ts_sentence_seq_labeling.py \
  TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/ts_sentence_seq_labeling.py \
    --model_name_or_path ${model_root_folder}/${model_name} \
    --dataset_name ./src/datasets/${dataset} \
    --dataset_cache_dir ${dataset_cache_dir} \
    --metric_name ${metric_name} \
    --gradient_checkpointing False \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --seed ${seed} \
    --max_seq_length ${max_seq_length} \
    --num_gpu ${NUM_GPU} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy steps \
    --eval_cnt ${num_train_epochs} \
    --load_best_model_at_end \
    --save_total_limit 2 \
    --metric_for_best_model overall_f1 \
    --eval_accumulation_steps 1000 \
    --overwrite_output_dir \
    --preprocessing_num_workers 5 \
    --ts_loss_weight ${ts_loss_weight} \
    --do_da_ts ${do_da_ts} \
    --do_cssl ${do_cssl} \
    --do_tssp ${do_tssp} \
    --cl_loss_weight ${cl_loss_weight} \
    --cl_temp ${cl_temp} \
    --cl_anchor_level ${cl_anchor_level} \
    --cl_positive_k ${cl_positive_k} \
    --cl_negative_k ${cl_negative_k} \
    --tssp_loss_weight ${tssp_loss_weight} \
    --output_dir ${OUTPUT_DIR} >> ${LOGFILE} 2>&1

  if [ $dataset = wiki_section ]; then
    echo "run postprocess_predictions to get_wiki_section_sent_level_metric"
    echo "then save sent level metric in "${LOGFILE}
    data_file=./data/$dataset/test.jsonl
    pred_file=${OUTPUT_DIR}/predict_wiki_section_max_seq${max_seq_length}_ts_score_lt.txt
    python ./src/postprocess_predictions.py $data_file $pred_file >> ${LOGFILE} 2>&1
  fi
done
