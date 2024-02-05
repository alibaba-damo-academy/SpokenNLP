#!/bin/bash

# base parameter
gpu_num=4
nshard=8

stage=0
stop_stage=7

n_cluster=2000
vocab_size=6000
down_sampling=1

corpus_dir=data/Librispeech
model_dir=data/models
dump_dir=data/dump/Librispeech
all_sets="test-clean test-other dev-clean dev-other train-clean-100 train-clean-360 train-other-500"
train_sets="train-clean-100 train-clean-360 train-other-500"
dev_sets="dev-clean dev-other"
test_sets="test-clean test-other"
train_sets_comma="train-clean-100,train-clean-360,train-other-500"
speed_factor_sets="1.0 0.9 1.1"
percent_data_kmeans=0.10416667

type="wavlm" # hubert or wavlm
if [ $type == "hubert" ]; then
  feat_dir=${dump_dir}/hubert_feat
  ckpt_path=${model_dir}/hubert_large_ll60k.pt
  layer=23
elif [ $type == "wavlm" ]; then
  feat_dir=${dump_dir}/wavlm_feat
  ckpt_path=${model_dir}/WavLM-Large.pt
  layer=23
fi

kmeans_model_path=${feat_dir}/kmeams.model
lab_combine_dir=${feat_dir}/label_combine
subword_model_path=${lab_combine_dir}/subword.model

# training parameter
num_processes=8 # default total batch size = 64, total_batch_size = num_precesses * per_device_train_batch_size
learning_rate=3e-4
max_epoch=10
mask_ratio=0.3
weight_kl_speech=0.008
weight_ce_speech=1
model_name=$(date +%Y%m%d%H)_gpt2_asr_model
output_dir=output/${model_name}

# Data downloading
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: data downloading"
  if [ ! -d ${corpus_dir} ]; then
    mkdir -p ${corpus_dir}
    echo "download training set to ${corpus_dir}"
    wget --no-check-certificate https://openslr.elda.org/resources/12/train-clean-100.tar.gz -P ${corpus_dir}/
    wget --no-check-certificate https://openslr.elda.org/resources/12/train-clean-360.tar.gz -P ${corpus_dir}/
    wget --no-check-certificate https://openslr.elda.org/resources/12/train-other-500.tar.gz -P ${corpus_dir}/

    echo "download dev set to ${corpus_dir}"
    wget --no-check-certificate https://openslr.elda.org/resources/12/dev-clean.tar.gz -P ${corpus_dir}/
    wget --no-check-certificate https://openslr.elda.org/resources/12/dev-other.tar.gz -P ${corpus_dir}/

    echo "download test set to ${corpus_dir}"
    wget --no-check-certificate https://openslr.elda.org/resources/12/test-clean.tar.gz -P ${corpus_dir}/
    wget --no-check-certificate https://openslr.elda.org/resources/12/test-other.tar.gz -P ${corpus_dir}/

    cd ${corpus_dir} || exit
    tar zxvf train-clean-100.tar.gz
    tar zxvf train-clean-360.tar.gz
    tar zxvf train-other-500.tar.gz
    tar zxvf dev-clean.tar.gz
    tar zxvf dev-other.tar.gz
    tar zxvf test-clean.tar.gz
    tar zxvf test-other.tar.gz
  fi
  if [ ! -d ${model_dir} ]; then
    mkdir -p ${model_dir}
    if [ $type == "hubert" ]; then
      wget --no-check-certificate https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt -P ${model_dir}/
    elif [ $type == "wavlm" ]; then
      echo "You can download WavLM-Large.pt manually from https://github.com/microsoft/unilm/blob/master/wavlm/README.md"
      exit
    fi
  fi

fi

# Prepare data
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: prepare data"
  mkdir -p ${dump_dir}
  for name in ${all_sets}; do
    echo "prepare ${name}"
    python fairseq/examples/wav2vec/wav2vec_manifest.py "${corpus_dir}/LibriSpeech/${name}" --dest ${dump_dir} --ext flac --valid-percent 0
    mv ${dump_dir}/train.tsv "${dump_dir}/${name}.tsv"
    python fairseq/examples/wav2vec/libri_labels.py "${dump_dir}/${name}.tsv" --output-dir ${dump_dir} --output-name "${name}"
  done

fi

# Extract features
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: extract features"
  mkdir -p ${feat_dir}
  for speed_factor in ${speed_factor_sets}; do
    if [ "$speed_factor" == "1.0" ]; then
      sets=${all_sets}
    else
      sets=${train_sets}
    fi
    for split in ${sets}; do
      for rank in $(seq 0 $((nshard - 1))); do
        feat_speed_perturbation_dir=${feat_dir}/feat_speed_perturbation_${speed_factor}
        let "remainder = ${rank} % ${gpu_num}"
        if [ $type == "hubert" ]; then
          CUDA_VISIBLE_DEVICES=${remainder} python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${dump_dir} \
            "${split}" ${ckpt_path} "${feat_speed_perturbation_dir}" ${layer} ${nshard} "${rank}" "${speed_factor}" &
        elif [ $type == "wavlm" ]; then
          CUDA_VISIBLE_DEVICES=${remainder} python fairseq/examples/hubert/simple_kmeans/dump_wavlm_feature.py ${dump_dir} \
            "${split}" ${ckpt_path} "${feat_speed_perturbation_dir}" ${layer} ${nshard} "${rank}" "${speed_factor}" &
        fi
      done
      wait
    done
  done
fi

# K-means clustering
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: K-means clustering"
  feat_speed_perturbation_dir=${feat_dir}/feat_speed_perturbation_1.0
  python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py ${feat_speed_perturbation_dir} ${train_sets_comma} ${nshard} ${kmeans_model_path} ${n_cluster} \
    --percent ${percent_data_kmeans} --reassignment_ratio 0.01 --batch_size 10000 --down_sampling ${down_sampling} 2>&1 | tee ${feat_speed_perturbation_dir}/log_learn_kmeans.txt
  wait
fi

# prepare label
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: prepare label"
  for speed_factor in ${speed_factor_sets}; do
    if [ "$speed_factor" == "1.0" ]; then
      sets=${all_sets}
    else
      sets=${train_sets}
    fi
    for split in ${sets}; do
      for rank in $(seq 0 $((nshard - 1))); do
        feat_speed_perturbation_dir=${feat_dir}/feat_speed_perturbation_${speed_factor}
        lab_dir=${feat_dir}/label_speed_perturbation_${speed_factor}
        mkdir -p "${lab_dir}"
        python fairseq/examples/hubert/simple_kmeans/dump_km.py "${feat_speed_perturbation_dir}" "${split}" \
          ${kmeans_model_path} ${nshard} "${rank}" "${lab_dir}" ${down_sampling} &
      done
      wait
    done
  done
fi

# postprocess data
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: postprocess data"
  for speed_factor in ${speed_factor_sets}; do
    if [ "$speed_factor" == "1.0" ]; then
      sets=${all_sets}
    else
      sets=${train_sets}
    fi
    lab_dir=${feat_dir}/label_speed_perturbation_${speed_factor}
    for split in ${sets}; do
      for rank in $(seq 0 $((nshard - 1))); do
        cat "${lab_dir}/${split}_${rank}_${nshard}.km"
      done >"${lab_dir}/${split}.km"
    done
  done

  ext=wrd
  for speed_factor in ${speed_factor_sets}; do
    for split in ${train_sets}; do
      cat "${dump_dir}/${split}.${ext}"
    done
  done >"${dump_dir}/train.${ext}"

  for split in ${dev_sets}; do
    cat "${dump_dir}/${split}.${ext}"
  done >"${dump_dir}/dev.${ext}"

  for split in ${test_sets}; do
    cat "${dump_dir}/${split}.${ext}"
  done >"${dump_dir}/test.${ext}"

  mkdir -p ${lab_combine_dir}
  ext=km
  for speed_factor in ${speed_factor_sets}; do
    lab_dir=${feat_dir}/label_speed_perturbation_${speed_factor}
    for split in ${train_sets}; do
      cat "${lab_dir}/${split}.${ext}"
    done
  done >"${lab_combine_dir}/train.${ext}"

  lab_dir=${feat_dir}/label_speed_perturbation_1.0
  for split in ${dev_sets}; do
    cat "${lab_dir}/${split}.${ext}"
  done >"${lab_combine_dir}/dev.${ext}"

  for split in ${test_sets}; do
    cat "${lab_dir}/${split}.${ext}"
  done >"${lab_combine_dir}/test.${ext}"

  for split in train dev test; do
    python utils/line_to_json.py --input_file_text "${dump_dir}/${split}.wrd" \
      --input_file_idx "${lab_combine_dir}/${split}.km" --output_file "${lab_combine_dir}/${split}.json"
  done
fi

# extract subword
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: extract subword"
  python utils/run_sentencepiece.py --input_file_train ${lab_combine_dir}/train.json --output_file_train ${lab_combine_dir}/train_subword.json \
    --input_file_validation ${lab_combine_dir}/dev.json --output_file_validation ${lab_combine_dir}/dev_subword.json \
    --input_file_test ${lab_combine_dir}/test.json --output_file_test ${lab_combine_dir}/test_subword.json \
    --model_prefix ${subword_model_path} --vocab_size ${vocab_size}
fi

# train and test model

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: train and test model"
  echo "$model_name"
  mkdir -p "${output_dir}"
  accelerate launch --multi_gpu --num_processes ${num_processes} --num_machines 1 --mixed_precision fp16 --dynamo_backend 'no'  transformers/examples/pytorch/language-modeling/run_clm.py \
    --train_file ${lab_combine_dir}/train_subword.json \
    --validation_file ${lab_combine_dir}/dev_subword.json \
    --test_file ${lab_combine_dir}/test_subword.json \
    --vocab_size_speech ${vocab_size} \
    --do_train \
    --do_predict \
    --model_name_or_path gpt2-medium \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=16 \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${max_epoch} \
    --block_size 1024 \
    --output_dir "${output_dir}" \
    --weight_kl_speech ${weight_kl_speech} \
    --weight_ce_speech ${weight_ce_speech} \
    --weight_ce_text 1.0 \
    --temperature 1 \
    --block_size 1024 \
    --max_text_length 150 \
    --predict_every_epoch \
    --overwrite_cache \
    --time_masking ${mask_ratio} \
    --report_to tensorboard 2>&1 | tee "${output_dir}/log_train.txt"

fi


