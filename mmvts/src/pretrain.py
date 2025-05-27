#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
import json
import time
import random

from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np

from tqdm import tqdm
from datasets import ClassLabel, load_dataset, load_metric

import evaluate
import transformers
import torch
import torch.optim as optim
import torch.nn.functional as F

torch.set_printoptions(profile="full")

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    BertTokenizerFast,
    LongformerTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)
from transformers import TrainerCallback, TrainerControl, TrainerState

from transformers.data.data_collator import (
    torch_default_data_collator,
    tf_default_data_collator,
    numpy_default_data_collator,
)


from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.integrations import TensorBoardCallback

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from types import SimpleNamespace

from arguments import *
from utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    
    if custom_args.test_data_name is None:
        custom_args.test_data_name = os.path.basename(data_args.dataset_name).split(".")[0]

    dataset_name = os.path.basename(data_args.dataset_name)
    model_name = abridge_model_name(model_args.text_encoder_name_or_path, model_args.vis2d_encoder_name)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    cur_file_path = os.path.abspath(__file__)
    data_dir = os.path.join(data_args.dataset_root_dir, dataset_name)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_dir=data_dir,
        cache_dir=data_args.dataset_cache_dir,
        ignore_verifications=True,
    )
    
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    features = raw_datasets["test"].features
    
    label_column_name = "labels"
    context_column_name = "sentences"
    example_id_column_name = "example_id"
    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name]) if "train" in raw_datasets else get_label_list(raw_datasets["test"][label_column_name])
        if data_args.task_name == "topic_segment":
            label_list = [_ for _ in label_list if _ != "-100"]
        label_to_id = {l: i for i, l in enumerate(label_list)}

    print("labels_are_int: ", labels_are_int)
    print("label_to_id: ", label_to_id)     # {'B-EOP': 0, 'O': 1}
    
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.text_encoder_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.update(model_args.__dict__)

    config.pretrain_task = custom_args.pretrain_task

    config.label_eot = label_to_id["B-EOP"]
    config.num_labels = num_labels
    config = update_config(config, model_args, data_args)

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.text_encoder_name_or_path
    if "longformer" in model_args.text_encoder_name_or_path:
        tokenizer = LongformerTokenizerFast.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.text_encoder_name_or_path,
            use_fast=True,
            use_auth_token=False,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
    
    pad_on_right = tokenizer.padding_side == "right"
    
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.max_seq_length > tokenizer.model_max_length:
        # resize position embeddings
        max_pos = data_args.max_seq_length
        config.max_position_embeddings = max_pos
        tokenizer.model_max_length = max_pos
        tokenizer.init_kwargs['model_max_length'] = tokenizer.model_max_length

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    config.max_seq_length = max_seq_length
    if config.language_type == "cn":
        max_seq_length -= 1     # longformer_zh 的 max_position_embeddings 是4096，所以position_ids最大只能是4095，而padding是0，所以input_ids只能有4095长度
    print("final max_seq_length: ", max_seq_length)

    if "longformer".lower() in model_args.text_encoder_name_or_path.lower():
        from models.multi_modal_for_ts import MultiModalForTS

        if config.pretrain_task == "ts":
            model = MultiModalForTS(config)
        else:
            raise ValueError("not supported pretrain_task {}".format(config.pretrain_task))
        
        if not model_args.init_model:
            model = model.from_pretrained(
                model_args.model_name_or_path,
                config = config,
                cache_dir = model_args.cache_dir,
                revision = model_args.model_revision,
                use_auth_token = True if model_args.use_auth_token else None,
            )
        print("model architecture: ", model)
    else:
        raise ValueError("%s not supported currently" % model_args.text_encoder_name_or_path.lower())

    if not tokenizer.bos_token:
        print("tokenizer has not bos_token. now add [BOS]")
        num_added_toks = tokenizer.add_special_tokens({'bos_token':'[BOS]'})
        model.text_encoder.text_encoder.resize_token_embeddings(len(tokenizer))
    else:
        print("tokenizer has bos_token {}, tokenizer.bos_token_id is {}".format(tokenizer.bos_token, tokenizer.bos_token_id))
        print("len(tokenizer): ", len(tokenizer))
        model.text_encoder.text_encoder.resize_token_embeddings(len(tokenizer))
    target_specical_ids = set()
    target_specical_ids.add(tokenizer.bos_token_id)
    
    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
                f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
            )
    print("label_list: ", label_list)

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def prepare_features_with_dynamic_num_sentence(examples):

        labels = examples[label_column_name]      # labels
        contexts = examples[context_column_name]        # sentences
        example_ids = examples[example_id_column_name]
        lectures = examples["lecture"]
        video_paths = examples["video_paths"]
        num_examples = len(example_ids)

        sentences = []
        for example_index in range(num_examples):
            example_sents = []
            for l, s in zip(labels[example_index], contexts[example_index]):
                add_suffix = tokenizer.bos_token
                example_sents.append(add_suffix + s)
            sentences.append(example_sents)
        
        try:
            tokenized_examples = tokenizer(
                sentences,
                is_split_into_words=True,
                add_special_tokens=False,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
        except Exception as e:
            print(str(e))
            print("++++ERROR++++")
            print(examples[example_id_column_name])
            print("++++ERROR++++")
            return {}
        
        # add token_seq_labels and other extra input field.
        token_seq_labels = []
        all_sent_bos_to_token_index = []    # record every example's sent_bos_to_token_index
        for example_index in range(num_examples):
            example_sent_labels = labels[example_index]
            example_sent_labels = [label_to_id[_] if _ in label_to_id else -100 for _ in example_sent_labels]
            example_input_ids = tokenized_examples["input_ids"][example_index]      # current input_ids have no cls token

            example_token_labels = []
            example_sent_bos_to_token_index = {}
            cur_sent_index = -1
            for token_index in range(len(example_input_ids)):
                if example_input_ids[token_index] in target_specical_ids:       # current token is bos
                    cur_sent_index += 1
                    example_token_labels.append(example_sent_labels[cur_sent_index])
                    example_sent_bos_to_token_index[cur_sent_index] = token_index
                else:
                    example_token_labels.append(-100)
            token_seq_labels.append(example_token_labels)
            all_sent_bos_to_token_index.append(example_sent_bos_to_token_index)
        
        tokenized_examples["token_seq_labels"] = token_seq_labels

        # slide-window to convert example to samples
        new_example_ids = []
        new_sentences = []
        new_token_seq_labels = []
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []

        new_vis_ids = []
        new_vis2d_embeds = []
        new_vis3d_embeds = []
        new_vis_ocr_embeds = []
        new_audio_embeds = []
        print("num_examples: ", num_examples)
        for example_index in tqdm(range(num_examples)):
            example_id = example_ids[example_index]
            lecture = lectures[example_index]
            example_video_paths = video_paths[example_index]        # used for loading clip feature
            print("example_id: {}, lecture: {}".format(example_id, lecture))

            example_sentences = [str(i) + "-" + s for i, s in enumerate(contexts[example_index])]
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_token_seq_labels = tokenized_examples["token_seq_labels"][example_index]
            example_token_type_ids = tokenized_examples["token_type_ids"][example_index]
            example_attention_mask = tokenized_examples["attention_mask"][example_index]

            # example_total_num_sentences = len(contexts[example_index])
            example_total_num_tokens = len(tokenized_examples["input_ids"][example_index])

            # the index of bos of every sentence in the whole token sequence
            example_sent_bos_to_token_index = all_sent_bos_to_token_index[example_index]

            # *tt*ttt*tttt (* means bos, t means token of sentence) => accumulate_legnth is [2, 6, 11]
            accumulate_length = [i - 1 for i in range(1, len(example_input_ids)) if example_input_ids[i] == tokenizer.bos_token_id] + [len(example_input_ids) - 1]
            
            token_left_index = 0    # [)
            sent_left_index = 0
            sent_i = 0
            while sent_i < len(accumulate_length):
                token_right_index = accumulate_length[sent_i] + 1
                sent_right_index = sent_i + 1
                if token_right_index - token_left_index >= max_seq_length - 1 or token_right_index == example_total_num_tokens:
                    
                    sample_input_ids = [tokenizer.cls_token_id] + example_input_ids[token_left_index:token_right_index]
                    sample_input_ids = sample_input_ids[:max_seq_length]

                    sample_token_seq_labels = [-100] + example_token_seq_labels[token_left_index:token_right_index]
                    sample_token_seq_labels = sample_token_seq_labels[:max_seq_length]

                    sample_token_type_ids = [0] + example_token_type_ids[token_left_index:token_right_index]
                    sample_token_type_ids = sample_token_type_ids[:max_seq_length]

                    sample_attention_mask = [1] + example_attention_mask[token_left_index:token_right_index]
                    sample_attention_mask = sample_attention_mask[:max_seq_length]

                    if sent_right_index - 1 == sent_left_index:
                        last_bos_index_in_sample = example_sent_bos_to_token_index[sent_left_index] - token_left_index + 1
                        sample_token_seq_labels[last_bos_index_in_sample] = -100
                        token_left_index = token_right_index
                    else:
                        last_bos_index_in_sample = example_sent_bos_to_token_index[sent_i] - token_left_index + 1
                        sample_token_seq_labels[last_bos_index_in_sample] = -100
                        token_left_index = accumulate_length[sent_i - 1] + 1      # neighboring windows share one sentence
                    
                    sample_sentences = example_sentences[sent_left_index:sent_right_index]

                    actual_sample_token_length = len(sample_input_ids)
                    while len(sample_input_ids) < max_seq_length:
                        sample_input_ids.append(tokenizer.pad_token_id)
                        sample_token_seq_labels.append(-100)
                        sample_token_type_ids.append(0)
                        sample_attention_mask.append(0)

                    new_example_ids.append(example_id)
                    new_sentences.append(sample_sentences)
                    new_input_ids.append(sample_input_ids)
                    new_token_seq_labels.append(sample_token_seq_labels)
                    new_token_type_ids.append(sample_token_type_ids)
                    new_attention_mask.append(sample_attention_mask)
                    
                    # cached features related
                    def load_cached_feature(example_video_paths, sent_left_index, sent_right_index, hidden_size, feature_type="2d"):
                        res = []
                        for clip_index in range(sent_left_index, sent_right_index):
                            clip_video_path = example_video_paths[clip_index]
                            clip_video_id = os.path.basename(clip_video_path).split(".mp4")[0]
                            if feature_type == "2d":
                                clip_feature_path = os.path.join(config.vis2d_feature_cache_dir, "{}.npy".format(clip_video_id))
                            elif feature_type == "3d":
                                clip_feature_path = os.path.join(config.vis3d_feature_cache_dir, "{}.npy".format(clip_video_id))
                            elif feature_type == "ocr":
                                clip_feature_path = os.path.join(config.vis_ocr_feature_cache_dir, "{}.npy".format(clip_video_id))
                            elif feature_type == "audio":
                                clip_feature_path = os.path.join(config.audio_feature_cache_dir, "{}.npy".format(clip_video_id))
                            else:
                                raise ValueError("not supported feature_type {}".format(feature_type))

                            if os.path.exists(clip_feature_path):
                                clip_emb = np.load(clip_feature_path)
                                if clip_emb.shape[0] == 0:
                                    print("clip_emb.shape[0] == 0")
                                    clip_emb = [0.0] * hidden_size
                                elif len(clip_emb.shape) == 2:
                                    if config.vis_embedding_pooling == "max":
                                        clip_emb = np.max(clip_emb, axis=0).tolist()
                                    elif config.vis_embedding_pooling == "mean":
                                        clip_emb = np.mean(clip_emb, axis=0).tolist()
                                    else:
                                        raise ValueError("not supported vis_embedding_pooling {}".format(config.vis_embedding_pooling))
                            else:
                                print("feature_type: {}, clip_index {}, clip_video_path: {}, clip_feature_path {} not exists.".format(feature_type, clip_index, clip_video_path, clip_feature_path))
                                clip_emb = [0.0] * hidden_size
                            res.append(clip_emb)

                        while len(res) < config.max_vis_seq_length:
                            res.append([0.0] * hidden_size)
                        if len(res) > config.max_vis_seq_length:
                            res = res[:config.max_vis_seq_length]
                        return res

                    cur_vis_ids = list(range(sent_left_index, sent_right_index))
                    while len(cur_vis_ids) < max_seq_length:
                        cur_vis_ids.append(-1)
                    new_vis_ids.append(cur_vis_ids)

                    if not config.use_raw_shot:
                        cur_vis_2d_embeds = load_cached_feature(example_video_paths, sent_left_index, sent_right_index, model_args.hidden_size_vis2d, feature_type="2d")
                        new_vis2d_embeds.append(cur_vis_2d_embeds)

                        cur_vis_3d_embeds = load_cached_feature(example_video_paths, sent_left_index, sent_right_index, model_args.hidden_size_vis3d, feature_type="3d")
                        new_vis3d_embeds.append(cur_vis_3d_embeds)

                        cur_vis_ocr_embeds = load_cached_feature(example_video_paths, sent_left_index, sent_right_index, model_args.hidden_size_vis_ocr, feature_type="ocr") 
                        new_vis_ocr_embeds.append(cur_vis_ocr_embeds)

                        cur_audio_embeds = load_cached_feature(example_video_paths, sent_left_index, sent_right_index, model_args.hidden_size_audio, feature_type="audio") 
                        new_audio_embeds.append(cur_audio_embeds)

                    if sent_right_index - 1 == sent_left_index or token_right_index == example_total_num_tokens:
                        sent_left_index = sent_right_index
                        sent_i += 1
                    else:
                        sent_left_index = sent_right_index - 1
                    
                else:
                    sent_i += 1

        output_samples = {
            "labels": new_token_seq_labels,
            "sentences": new_sentences,
            "input_ids": new_input_ids,
            "token_type_ids": new_token_type_ids,
            "attention_mask": new_attention_mask,
            "example_ids": new_example_ids,
            "vis_ids": new_vis_ids, # combine example_ids, vis_ids and current_mode to get image path
        }
        if not config.use_raw_shot:
            output_samples["vis2d_embeds"] = new_vis2d_embeds
            output_samples["vis3d_embeds"] = new_vis3d_embeds
            output_samples["vis_ocr_embeds"] = new_vis_ocr_embeds
            output_samples["audio_embeds"] = new_audio_embeds

        return output_samples

    train_preprocess_fn = prepare_features_with_dynamic_num_sentence
    valid_preprocess_fn = prepare_features_with_dynamic_num_sentence
    test_preprocess_fn = prepare_features_with_dynamic_num_sentence

    ab_dataset_name = abridge_dataset_name(dataset_name)
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                train_preprocess_fn,
                batched=True,
                batch_size=10,
                remove_columns=remove_columns,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, ab_dataset_name, max_seq_length, "train"),
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        total_train_samples = len(train_dataset)
        if model_args.num_gpu == 0:
            model_args.num_gpu = 1      # use cpu 
        total_train_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * model_args.num_gpu
        total_train_steps = total_train_samples * training_args.num_train_epochs // total_train_batch_size
        eval_steps = total_train_steps // data_args.eval_cnt
        eval_steps = max(eval_steps, 40)
        training_args.logging_steps = 5
        training_args.eval_steps = eval_steps
        training_args.save_steps = eval_steps
        print("eval_steps: ", eval_steps)

        # for i in range(1):
        #     print("train sample %d: " % i)
        #     print(train_dataset[i])

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                valid_preprocess_fn,
                batched=True,
                batch_size=10,
                remove_columns=remove_columns,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, ab_dataset_name, max_seq_length, "dev"),
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        # for i in range(1):
        #     print("dev sample {}: {}".format(i, eval_dataset[i]))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_examples), data_args.max_predict_samples)
            predict_examples = predict_examples.select(range(max_predict_samples))
        
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                test_preprocess_fn,
                batched=True,
                batch_size=10,
                remove_columns=remove_columns,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, ab_dataset_name, max_seq_length, "test"),
                desc="Running tokenizer on prediction dataset",
            )
    
    data_collator = default_data_collator

    def select_features(features, labels, example_level=False):
        example_num = labels.shape[0]
        mask = labels != -100
        valid_features_count = mask.sum(1)
        # print("valid_features_count: ", valid_features_count)
        selected_features_list = [features[i, :valid_features_count[i]] for i in range(example_num)]
        if not example_level:
            selected_features = torch.cat([torch.tensor(v) for v in selected_features_list], dim=0)
            return selected_features
        else:
            return selected_features_list        
    
    # Metrics
    metric = load_metric(data_args.metric_name)
    def compute_metrics(p):
        if config.pretrain_task == "ts":
            predictions_logits, labels = p
            true_labels = [[label_list[l] for l in label if l != -100] for label in labels]

            predictions = np.argmax(predictions_logits, axis=2)     # 0 will be converted to B-EOP by label_list, so 0 means segmentation point
            true_predictions = [
                [label_list[prediction[i]] for i in range(len(label))]  # prediction[:k] 是有效的logits，k是label（一个seq样本的标签）中非-100的数目
                for prediction, label in zip(predictions, true_labels)
            ]
            results = metric.compute(predictions=true_predictions, references=true_labels)
            
            if data_args.return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }
        elif config.pretrain_task == "align":
            features, labels = p
            text_features, vis_features = features[0], features[1]
            # print("labels.shape: ", labels.shape)
            # print("text_features.shape: ", text_features.shape)
            sel_text_features = select_features(text_features, labels)
            sel_vis_features = select_features(vis_features, labels)
            cosine_sim = torch.mean(F.cosine_similarity(sel_text_features, sel_vis_features, dim=-1, eps=1e-8))
            return {
                "cosine_sim": cosine_sim.item()
            }
        else:
            raise ValueError("not supported pretrain_task {}".format(pretrain_task))


    # class ModeSettingCallback(TrainerCallback):
    #     def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #         kwargs['model'].set_mode('train')

    #     def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #         kwargs['model'].set_mode('dev')

    #     def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #         kwargs['model'].set_mode('test')

    # Initialize our Trainer
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 初始化一个变量用来存储辅助损失
            self.loss_dict = None

        def compute_loss(self, model, inputs, return_outputs=False):          
            # 此处我们注册一个 hook，以便捕获模型内部生成的辅助损失
            # 你需要找到产生辅助损失的模型层的名称，这里我们假设它是 'compute_aux_loss'
            def hook_function(module, input, output):
                # 假设辅助损失在输出中的名字是 'loss'
                self.loss_dict = output['loss']
            
            # 检查模型是否被包装在 DataParallel 或 DistributedDataParallel 中
            unwrapped_model = model.module if hasattr(model, 'module') else model
            # 注册 hook
            handle = unwrapped_model.predictor.predictor.loss_layer.register_forward_hook(hook_function)
            
            # 执行损失计算并移除 hook
            result = super().compute_loss(model, inputs, return_outputs)
            handle.remove()

            # 记录辅助损失
            if self.loss_dict is not None:
                for k, v in self.loss_dict.items():
                    self.log({k: v.item()})
            
            return result
    
    # 设置不同学习率的特定模块
    cross_encoder_module = model.cross_encoder
    cross_encoder_params = {'params': cross_encoder_module.parameters(), 'lr': config.cross_encoder_lr}

    # 记得先把特定模块的参数剔除为后续准备
    def filter_fct(p):
        params_list = list(map(id, cross_encoder_module.parameters()))
        return id(p) not in params_list
    base_params = filter(filter_fct, model.parameters())

    optimizer = optim.Adam([
        {'params': base_params, 'lr': training_args.learning_rate},  # 默认学习率
        cross_encoder_params,
    ])

    if config.pretrain_task == "ts":
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, None),
            compute_metrics=compute_metrics,
            # callbacks=[ModeSettingCallback, TensorBoardCallback],
            callbacks=[TensorBoardCallback],
        )
    else:
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        # callbacks=[ModeSettingCallback, TensorBoardCallback],
        callbacks=[TensorBoardCallback],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict and config.pretrain_task == "ts":
        metric_key_prefix = "predict"
        logger.info("*** Predict test set ***")
        sentences = predict_dataset["sentences"]
        example_ids = predict_dataset["example_ids"]
        num_examples = len(predict_examples[context_column_name])
        num_samples = len(predict_dataset["sentences"])
        print("num predict samples: ", num_samples)
        print("num predict examples: ", num_examples)

        start_time = time.time()
        predict_logits, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix=metric_key_prefix)
        end_time = time.time()
        print("predict_time(s): ", end_time - start_time)
        
        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        true_int_labels = [[l for l in label if l != -100] for label in labels]

        predictions = np.argmax(predict_logits, axis=2)

        true_predictions = [
            [label_list[prediction[i]] for i in range(len(label))]  # prediction[:k] 是有效的logits，k是label（一个seq样本的标签）中非-100的数目
            for prediction, label in zip(predictions, true_labels)
        ]
        true_predict_logits = [
            [p_logits[i].tolist() for i in range(len(label))]
            for p_logits, label in zip(predict_logits, true_labels)
        ]

        metric_file_name = "_".join([metric_key_prefix, custom_args.test_data_name, "max_seq%d" % data_args.max_seq_length])
        trainer.log_metrics(metric_file_name, metrics)
        trainer.save_metrics(metric_file_name, metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, metric_file_name + ".txt")
        out = [{"example_id": [], "sentences": [], "labels": [], "int_labels": [], "predictions": [], "predict_logits": []} for _ in range(num_examples)]
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction, sentence_list, label, int_labels, example_id, predict_logits in zip(
                    true_predictions, sentences, true_labels, true_int_labels, example_ids, true_predict_logits):
                    out[example_id]["example_id"].append(example_id)
                    out[example_id]["sentences"].extend(sentence_list)      # TODO overlap sentences
                    out[example_id]["labels"].extend(label)
                    out[example_id]["predictions"].extend(prediction)
                    out[example_id]["predict_logits"].extend(predict_logits)
                    out[example_id]["int_labels"].extend([int(v) for v in int_labels])

                writer.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in out])
            
            example_level_predictions_logits = [v["predict_logits"] for v in out]
            example_level_int_labels = [v["int_labels"] for v in out]
            example_level_str_labels = [v["labels"] for v in out]                
            
            print("call compute_metric_example_level()")
            start_time = time.time()
            print("start_time: ", start_time)
            
            # filter empty examples which have just one sentence
            logits_not_empty, labels_not_empty, labels_str_not_empty = [], [], []
            for exp_logits, exp_labels, exp_labels_str in zip(example_level_predictions_logits, example_level_int_labels, example_level_str_labels):
                if len(exp_labels) != 0:
                    assert len(exp_logits) != 0
                    logits_not_empty.append(exp_logits)
                    labels_not_empty.append(exp_labels)
                    labels_str_not_empty.append(exp_labels_str)
            
            # compute example metric
            example_level_metric = metric.compute_metric_example_level(logits_not_empty, labels_not_empty, label_list, custom_args, data_args, mode=metric_key_prefix)
            end_time = time.time()
            print("end_time: ", end_time)
            print("time used(s): ", end_time - start_time)
            example_level_metric["%s_examples" % metric_key_prefix] = num_examples
            print("example_level_metric: ", example_level_metric)
            example_level_metric_file_name = "example_level_" + metric_file_name
            trainer.log_metrics(example_level_metric_file_name, example_level_metric)
            trainer.save_metrics(example_level_metric_file_name, example_level_metric)

            res_file_path = os.path.join(training_args.output_dir, example_level_metric_file_name + "_results.json")
            convert_res_format(res_file_path, custom_args)

            print("done")
    
    if training_args.do_predict and config.pretrain_task == "align":
        metric_key_prefix = "predict"
        logger.info("*** Predict test set ***")
        sentences = predict_dataset["sentences"]
        example_ids = predict_dataset["example_ids"]
        num_examples = len(predict_examples[context_column_name])
        num_samples = len(predict_dataset["sentences"])
        print("num predict samples: ", num_samples)
        print("num predict examples: ", num_examples)

        start_time = time.time()
        features, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix=metric_key_prefix)
        end_time = time.time()
        print("predict_time(s): ", end_time - start_time)
        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        true_int_labels = [[l for l in label if l != -100] for label in labels]

        text_features, vis_features = features[0], features[1]
        sel_text_features = select_features(text_features, labels, example_level=True)
        sel_vis_features = select_features(vis_features, labels, example_level=True)
        # print("labels: ", labels)
        # print("sel_text_features: ", sel_text_features)

        metric_file_name = "_".join([metric_key_prefix, custom_args.test_data_name, "max_seq%d" % data_args.max_seq_length])
        trainer.log_metrics(metric_file_name, metrics)
        trainer.save_metrics(metric_file_name, metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, metric_file_name + ".txt")
        out = [{"example_id": [], "sentences": [], "labels": [], "int_labels": [], "text_feature": [], "vis_feature": []} for _ in range(num_examples)]
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for sentence_list, label, int_labels, example_id, text_feature, vis_feature in zip(
                    sentences, true_labels, true_int_labels, example_ids, sel_text_features, sel_vis_features):
                    out[example_id]["example_id"].append(example_id)
                    out[example_id]["sentences"].extend(sentence_list)      # TODO overlap sentences
                    out[example_id]["labels"].extend(label)
                    out[example_id]["int_labels"].extend([int(v) for v in int_labels])
                    out[example_id]["text_feature"].extend(text_feature.tolist())
                    out[example_id]["vis_feature"].extend(vis_feature.tolist())

                writer.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in out])

            print("done")



if __name__ == "__main__":
    main()