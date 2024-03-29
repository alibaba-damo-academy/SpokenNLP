export CUDA_VISIBLE_DEVICES=0,1
NUM_GPU=2
echo $CUDA_VISIBLE_DEVICES
echo $NUM_GPU

PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

metric_name=./src/metrics/seqeval.py

# max_seq_length=512
max_seq_length=2048

# model_name=bert_base_uncased
# model_name=electra
# model_name=bigbird_roberta_base

model_name=longformer
model_path=path/to/the/finetuned/model

# dataset=wiki_section
# dataset=wiki_section_disease
# dataset=wiki_section_city
# dataset=wiki727k
# dataset=wiki50
# dataset=wiki_elements

for dataset in wiki_section; do
  dataset_cache_dir=./cached_data/${dataset}_${model_name}_${max_seq_length}
  LOGFILE=$model_path/infer.log
  echo "write log into "${LOGFILE}
  echo $CUDA_VISIBLE_DEVICES >> ${LOGFILE}

  # python ./src/ts_sentence_seq_labeling.py \
  TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/ts_sentence_seq_labeling.py \
    --model_name_or_path ${model_path} \
    --dataset_name ./src/datasets/${dataset} \
    --dataset_cache_dir ${dataset_cache_dir} \
    --metric_name ${metric_name} \
    --gradient_checkpointing False \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --max_seq_length ${max_seq_length} \
    --num_gpu ${NUM_GPU} \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir False \
    --preprocessing_num_workers 5 \
    --output_dir ${model_path} >> ${LOGFILE} 2>&1

  if [ $dataset = wiki_section ]; then
    echo "run postprocess_predictions to get_wiki_section_sent_level_metric"
    echo "then save sent level metric in "${LOGFILE}
    data_file=./data/$dataset/test.jsonl
    pred_file=${model_path}/predict_wiki_section_max_seq${max_seq_length}_ts_score_lt.txt
    python ./src/postprocess_predictions.py $data_file $pred_file >> ${LOGFILE} 2>&1
  fi
done