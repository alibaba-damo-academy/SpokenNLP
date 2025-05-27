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

vis2d_encoder_name=cn_clip_vit_b_16
hidden_size_vis2d=512

audio_encoder_name=whisper_small
hidden_size_audio=768

use_raw_shot=False    # cache vis features
# use_raw_shot=True     # not cache vis features
vis2d_feature_cache_dir=/path/to/vis_2d_feature_embeddings
vis3d_feature_cache_dir=/path/to/vis_3d_feature_embeddings
vis_ocr_feature_cache_dir=/path/to/vis_ocr_feature_embeddings
audio_feature_cache_dir=/path/to/vis_audio_feature_embeddings

dataset=clvts
dataset_root_dir=/absolute/path/to/the/dataset/root/dir/
dataset_cache_dir=./cached_data/${dataset}-text_${text_encoder_name}-vis2d_${vis2d_encoder_name}-ocr_sbert-${max_seq_length}-row_shot_${use_raw_shot}

use_vis2d=True
use_vis3d=True
use_vis_ocr=True
use_audio=True

freeze_text_encoder=False
freeze_vis2d_encoder=True

######## cross_encoder ########
cross_encoder_type=ca_moe   # self-attn + cross-attn + moe
num_cross_encoder_layers=1
num_cross_encoder_heads=12
cross_encoder_lr=5e-5

######## predictor ########
fuse_type=cat

######## contrastive learning ########
do_modality_cl=True
modality_cl_lw=0.5

do_align_av=True
align_av_weight=0.1

do_align_at=True
align_at_weight=0.1

do_align_tv=True
align_tv_weight=0.8

topic_cl_type=list
topic_cl_choice=near
topic_cl_fct=simcse

do_topic_mm_cl=False
topic_mm_cl_lw=0.5
topic_mm_cl_pos_k=1
topic_mm_cl_neg_k=3

seed=42
lr=5e-5
num_train_epochs=5
per_device_train_batch_size=1
gradient_accumulation_steps=$(( 8 / NUM_GPU))

weight_label_zero=0.7
num_cross_encoder_layers=1

for fuse_type in cat; do
for cross_encoder_type in ma_moe; do
for seed in 42 59 88; do

  bs=`expr $NUM_GPU \* $per_device_train_batch_size \* $gradient_accumulation_steps`

  currentDate=`date "+%Y-%m-%d"`
  currentTime=`date "+%Y-%m-%d_%H:%M:%S"`
  output_dir=./output/finetune-${dataset}-t_${text_encoder_name}-v2d_${vis2d_encoder_name}-audio_${audio_encoder_name}-cross_type_${cross_encoder_type}_l${num_cross_encoder_layers}_h${num_cross_encoder_heads}_lr${cross_encoder_lr}/seq${max_seq_length}-wlz${weight_label_zero}-lr${lr}-epoch${num_train_epochs}-bs${bs}/seed${seed}-fuse_${fuse_type}-vis2d_${use_vis2d}-vis3d_${use_vis3d}-ocr_${use_vis_ocr}/mcl_${do_modality_cl}_${modality_cl_lw}_av${do_align_av}${align_av_weight}_at${do_align_at}${align_at_weight}_vt${do_align_tv}${align_tv_weight}-do_topic_mm_cl_${do_topic_mm_cl}_${topic_mm_cl_lw}_p${topic_mm_cl_pos_k}_n${topic_mm_cl_neg_k}-${currentTime}

  mkdir -p $output_dir
  log_file=$output_dir/run.log
  echo "write log into "${log_file}
  echo $CUDA_VISIBLE_DEVICES >> ${log_file}

  export CUDA_LAUNCH_BLOCKING=1
  TORCH_DISTRIBUTED_DEBUG=DETAIL ${python} -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ./src/main_multimodal.py \
    --language_type cn \
    --use_raw_shot ${use_raw_shot} \
    --text_encoder_name_or_path ${model_root_folder}/${text_encoder_name} \
    --vis2d_encoder_name ${vis2d_encoder_name} \
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
    --use_vis2d ${use_vis2d} \
    --use_vis3d ${use_vis3d} \
    --use_vis_ocr ${use_vis_ocr} \
    --hidden_size_vis2d ${hidden_size_vis2d} \
    --vis2d_feature_cache_dir ${vis2d_feature_cache_dir} \
    --vis3d_feature_cache_dir ${vis3d_feature_cache_dir} \
    --vis_ocr_feature_cache_dir ${vis_ocr_feature_cache_dir} \
    --audio_feature_cache_dir ${audio_feature_cache_dir} \
    --hidden_size_audio ${hidden_size_audio} \
    --weight_label_zero ${weight_label_zero} \
    --freeze_text_encoder ${freeze_text_encoder} \
    --freeze_vis2d_encoder ${freeze_vis2d_encoder} \
    --cross_encoder_type ${cross_encoder_type} \
    --num_cross_encoder_layers ${num_cross_encoder_layers} \
    --num_cross_encoder_heads ${num_cross_encoder_heads} \
    --cross_encoder_lr ${cross_encoder_lr} \
    --fuse_type ${fuse_type} \
    --do_modality_cl ${do_modality_cl} \
    --do_align_av ${do_align_av} \
    --align_av_weight ${align_av_weight} \
    --do_align_at ${do_align_at} \
    --align_at_weight ${align_at_weight} \
    --do_align_tv ${do_align_tv} \
    --align_tv_weight ${align_tv_weight} \
    --modality_cl_lw ${modality_cl_lw} \
    --do_topic_mm_cl ${do_topic_mm_cl} \
    --topic_mm_cl_lw ${topic_mm_cl_lw} \
    --topic_mm_cl_pos_k ${topic_mm_cl_pos_k} \
    --topic_mm_cl_neg_k ${topic_mm_cl_neg_k} \
    --topic_cl_type ${topic_cl_type} \
    --topic_cl_choice ${topic_cl_choice} \
    --topic_cl_fct ${topic_cl_fct} \
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
done
done
