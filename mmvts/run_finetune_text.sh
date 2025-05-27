export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8

echo $CUDA_VISIBLE_DEVICES
echo $NUM_GPU
python=~/anaconda3/envs/torch1.12.1/bin/python

PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

metric_name=./src/metrics/seqeval.py
model_root_folder=./pretrained_models

max_seq_length=2048
# max_seq_length=4096

text_encoder_name=longformer_zh

dataset=clvts
dataset_root_dir=/absolute/path/to/the/dataset/root/dir/
dataset_cache_dir=./cached_data/${dataset}-text_${text_encoder_name}

seed=42
lr=5e-5
num_train_epochs=5
per_device_train_batch_size=1
gradient_accumulation_steps=$(( 8 / NUM_GPU))

weight_label_zero=0.7
fuse_type=text_only
for seed in 42 59 88; do

  bs=`expr $NUM_GPU \* $per_device_train_batch_size \* $gradient_accumulation_steps`

  currentDate=`date "+%Y-%m-%d"`
  currentTime=`date "+%Y-%m-%d_%H:%M:%S"`
  output_dir=./output/finetune-${dataset}-t_${text_encoder_name}/seq${max_seq_length}-wlz${weight_label_zero}-lr${lr}-epoch${num_train_epochs}-bs${bs}/seed${seed}-fuse_${fuse_type}-${currentTime}

  mkdir -p $output_dir
  log_file=$output_dir/run.log
  echo "write log into "${log_file}
  echo $CUDA_VISIBLE_DEVICES >> ${log_file}

  export CUDA_LAUNCH_BLOCKING=1
  TORCH_DISTRIBUTED_DEBUG=DETAIL ${python} -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/main_text.py \
    --language_type cn \
    --text_encoder_name_or_path ${model_root_folder}/${text_encoder_name} \
    --init_model True \
    --dataset_name ./src/datasets/${dataset} \
    --dataset_root_dir ${dataset_root_dir} \
    --dataset_cache_dir ${dataset_cache_dir} \
    --metric_name ${metric_name} \
    --gradient_checkpointing False \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --seed ${seed} \
    --weight_label_zero ${weight_label_zero} \
    --fuse_type ${fuse_type} \
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
    --preprocessing_num_workers 1 \
    --output_dir ${output_dir} >> ${log_file} 2>&1

  test_data_file=${dataset_root_dir}/${dataset}/test.jsonl
  test_pred_file=${output_dir}/predict_${dataset}_max_seq${max_seq_length}.txt
  test_metric_file=$output_dir/example_level_predict_${dataset}_max_seq${max_seq_length}_results_str_metric.txt
  ${python} ./src/evaluate.py -d ${test_data_file} -p ${test_pred_file} -l cn >> ${test_metric_file}
done
