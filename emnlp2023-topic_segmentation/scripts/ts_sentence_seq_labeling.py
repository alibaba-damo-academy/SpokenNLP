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
from datasets import ClassLabel, load_dataset, load_metric

import evaluate
import transformers
import torch
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
    model_name = abridge_model_name(model_args.model_name_or_path)

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
    data_dir = os.path.join(os.path.dirname(cur_file_path), "../data/{}".format(dataset_name))        
    
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
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.update(model_args.__dict__)
    config.num_tssp_labels = 3
    print("config: ", config)
    
    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    elif "longformer" in model_args.model_name_or_path:
        tokenizer = LongformerTokenizerFast.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
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

    model_type = None
    if "longformer".lower() in model_args.model_name_or_path.lower():
        model_type = "longformer"
        from models.longformer_for_ts import LongformerWithDAForSentenceLabelingTopicSegmentation
        model = LongformerWithDAForSentenceLabelingTopicSegmentation.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            cache_dir = model_args.cache_dir,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
    elif "bigbird".lower() in model_args.model_name_or_path.lower():
        model_type = "bigbird"
        from models.bigbird_for_ts import BigBirdWithDAForSentenceLabelingTopicSegmentation
        model = BigBirdWithDAForSentenceLabelingTopicSegmentation.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            cache_dir = model_args.cache_dir,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
    elif "bert" in model_args.model_name_or_path.lower():
        model_type = "bert"
        from models.bert_for_ts import BertWithDAForSentenceLabelingTopicSegmentation
        model = BertWithDAForSentenceLabelingTopicSegmentation.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            cache_dir = model_args.cache_dir,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=True,
        )
    elif "electra" in model_args.model_name_or_path.lower():
        model_type = "electra"
        from models.electra_for_ts import ElectraWithDAForSentenceLabelingTopicSegmentation
        model = ElectraWithDAForSentenceLabelingTopicSegmentation.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            cache_dir = model_args.cache_dir,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError("%s not supported currently" % model_args.model_name_or_path.lower())

    print("model_type: ", model_type)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
    
    pad_on_right = tokenizer.padding_side == "right"
    if not tokenizer.bos_token:
        num_added_toks = tokenizer.add_special_tokens({'bos_token':'[BOS]'})
        model.resize_token_embeddings(len(tokenizer))
    target_specical_ids = set()
    target_specical_ids.add(tokenizer.bos_token_id)
    
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
    print("final max_seq_length: ", max_seq_length)

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

    def get_extract_eop_segment_ids(sample_input_ids, sample_token_seq_labels):
        extract_eop_segment_ids = [0]
        eop_id = 1
        for i in range(1, len(sample_input_ids)):
            input_id = sample_input_ids[i]
            if input_id in target_specical_ids:
                if sample_token_seq_labels[i] != -100:
                    extract_eop_segment_ids.append(eop_id)
                    eop_id += 1
                else:
                    extract_eop_segment_ids.append(0)
            else:
                extract_eop_segment_ids.append(0)
        return extract_eop_segment_ids
    
    def get_sample_sent_token_mask(sample_input_ids, sample_token_seq_labels):
        # for extracting bos token and corresponding topic_segment_ids. format is like [-100, 1, -100, .., 0, -100, ..., 1, -100, ...] where 1 or 0 is bos token and 0 means end sentence of topic
        sample_sent_token_mask = [-100]  # first -100 is for cls
        for i in range(1, len(sample_input_ids)):
            input_id = sample_input_ids[i]
            token_label = sample_token_seq_labels[i]
            if input_id in target_specical_ids:
                if token_label == 0:
                    sample_sent_token_mask.append(0)
                else:
                    sample_sent_token_mask.append(1)
            else:
                sample_sent_token_mask.append(-100)
        return sample_sent_token_mask

    def shuffle_and_replace_doc_topics(
        tokenized_examples,
        num_examples,
        example_index,
        example_sentences,
        example_sent_labels,
        example_sent_index_to_start_end_token_index,
        example_input_ids,
        topic_start_sent_indices, 
        topic_end_sent_indices,
        all_topic_index_to_start_end_sent_index, 
        all_sent_index_to_sent_text,
        all_sent_label,
        all_sent_index_to_start_end_token_index,
        ):

        da_example_input_ids = []
        da_example_sentences = []
        da_example_sent_label_ids = []
        da_example_neighbor_sent_pair_order_labels = []
        replaced_flag = False
        topic_orders = []   # if replace, then the value is -1, else the value is original topic-level order in doc.

        # shuffle topics in doc, then replace some topics by probability p1=0.5 and p2=0.5
        topic_cnt = len(topic_start_sent_indices)
        topic_indices = list(range(topic_cnt))
        random.shuffle(topic_indices)
        topic_orders = [v for v in topic_indices]
        p1 = random.random()
        if p1 > 0.5 and num_examples > 1:
            # replace some topics randomly
            for i, topic_index in enumerate(topic_indices):
                p2 = random.random()
                if p2 > 0.5:
                    replaced_flag = True
                    topic_orders[i] = -1
                    # replace topic with topic of other examples
                    example_choices = list(range(num_examples))
                    example_choices.remove(example_index)
                    random_example_index = random.choice(example_choices)
                    
                    new_topic_index_choices = list(range(len(all_topic_index_to_start_end_sent_index[random_example_index])))
                    random_topic_index_from_other_examples = random.choice(list(range(len(new_topic_index_choices))))

                    new_topic_start_sent_index, new_topic_end_sent_index = all_topic_index_to_start_end_sent_index[random_example_index][random_topic_index_from_other_examples]
                    for s_id in range(new_topic_start_sent_index, new_topic_end_sent_index + 1):
                        da_example_sentences.append(str(random_example_index) + "-" + str(random_topic_index_from_other_examples) + "-" + str(s_id) + "-" + all_sent_index_to_sent_text[random_example_index][s_id])
                        da_example_sent_label_ids.append(all_sent_label[random_example_index][s_id])
                        sent_start_token_index, sent_end_token_index = all_sent_index_to_start_end_token_index[random_example_index][s_id]
                        random_sent_input_ids = tokenized_examples["input_ids"][random_example_index][sent_start_token_index : sent_end_token_index + 1]
                        da_example_input_ids += random_sent_input_ids

                        # get da_example_neighbor_sent_pair_order_labels
                        if s_id == new_topic_start_sent_index:
                            da_example_neighbor_sent_pair_order_labels.append(2)    # start sentence of new topic
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(0)
                        sent_b_token_cnt = sent_end_token_index - sent_start_token_index + 1
                        da_example_neighbor_sent_pair_order_labels += [-100 for _ in range(sent_b_token_cnt - 1)]
                else:
                    cur_topic_start_sent_index = topic_start_sent_indices[topic_index]
                    cur_topic_end_sent_index = topic_end_sent_indices[topic_index]
                    for s_id in range(cur_topic_start_sent_index, cur_topic_end_sent_index + 1):
                        da_example_sentences.append(str(topic_index) + "-" + str(s_id) + "-" + example_sentences[s_id])
                        da_example_sent_label_ids.append(example_sent_labels[s_id])
                        sent_start_token_index, sent_end_token_index = example_sent_index_to_start_end_token_index[s_id]
                        da_example_input_ids += example_input_ids[sent_start_token_index : sent_end_token_index + 1]

                        # get da_example_neighbor_sent_pair_order_labels
                        if s_id == cur_topic_start_sent_index:
                            da_example_neighbor_sent_pair_order_labels.append(2)    # start sentence of new topic
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(0)
                        sent_b_token_cnt = sent_end_token_index - sent_start_token_index + 1
                        da_example_neighbor_sent_pair_order_labels += [-100 for _ in range(sent_b_token_cnt - 1)]
        else:
            for topic_index in topic_indices:
                new_topic_start_sent_index = topic_start_sent_indices[topic_index]
                new_topic_end_sent_index = topic_end_sent_indices[topic_index]
                for s_id in range(new_topic_start_sent_index, new_topic_end_sent_index + 1):
                    da_example_sentences.append(str(topic_index) + "-" + str(s_id) + "-" + example_sentences[s_id])
                    da_example_sent_label_ids.append(example_sent_labels[s_id])
                    sent_start_token_index, sent_end_token_index = example_sent_index_to_start_end_token_index[s_id]
                    da_example_input_ids += example_input_ids[sent_start_token_index : sent_end_token_index + 1]

                    # get da_example_neighbor_sent_pair_order_labels
                    if s_id == new_topic_start_sent_index:
                        da_example_neighbor_sent_pair_order_labels.append(2)    # start sentence of new topic
                    else:
                        da_example_neighbor_sent_pair_order_labels.append(0)
                    sent_b_token_cnt = sent_end_token_index - sent_start_token_index + 1
                    da_example_neighbor_sent_pair_order_labels += [-100 for _ in range(sent_b_token_cnt - 1)]

        return da_example_input_ids, da_example_sentences, da_example_sent_label_ids, da_example_neighbor_sent_pair_order_labels, replaced_flag, topic_orders

    def shuffle_topic_sents(
        example_input_ids,
        example_sentences,
        example_sent_labels,
        example_sent_index_to_start_end_token_index,
        topic_start_sent_indices,
        topic_end_sent_indices,
        tssp_ablation="none",
        topic_orders=None,
    ):
        # choices for tssp_ablation are ["none", "wo_intra_topic", "wo_inter_topic", "sso"]
        # none is for 0,1,2 where 0 means b is nsp of a and in the same topic; 1 means b is not nsp of a but in the same topic; 2 means b is start sentence of new topic.
        # wo_intra_topic means there just exists two class, where 0 means b and a are in the same topic, 1 means b and a are not in the same topic
        # wo_inter_topic means there just exists two class, where 0 means b is nsp of a, and 1 means b is not nsp of a. just like bert.
        # sso is for implementing struct-bert. where 0 means Next Sent Prediction; 1 means Prev. Sent Prediction; 2 means Random Sent Prediction.
        da_example_input_ids = []
        da_example_sentences = []
        da_example_sent_label_ids = []
        da_example_neighbor_sent_pair_order_labels = []
        replaced_flag = False

        for i in range(len(topic_start_sent_indices)):
            start_index = topic_start_sent_indices[i]
            end_index = topic_end_sent_indices[i]
            topic_sentences = example_sentences[start_index : end_index + 1]

            # shuffle at sentence input ids aspect. then get sentence info (example id and sentece id) 
            topic_sent_indices = list(range(start_index, end_index))
            random.shuffle(topic_sent_indices)
            topic_sent_indices.append(end_index)        # always keep eot sentence as eot sentence

            new_topic_sent_label_ids = [label_to_id["O"]] * (len(topic_sent_indices) - 1) + [label_to_id["B-EOP"]]
            old_topic_sent_label_ids = []
            for j, sent_index in enumerate(topic_sent_indices):
                # value of sent_index is in range of example_sentences
                sent_start_token_index, sent_end_token_index = example_sent_index_to_start_end_token_index[sent_index]
                da_example_input_ids += example_input_ids[sent_start_token_index : sent_end_token_index + 1]
                da_example_sentences.append(str(sent_index) + "-" + example_sentences[sent_index])
                old_topic_sent_label_ids.append(example_sent_labels[sent_index])

                if tssp_ablation == "none":
                    if j == 0:
                        da_example_neighbor_sent_pair_order_labels.append(2)    # start sentence of new topic
                    else:
                        sent_a_index = topic_sent_indices[j - 1]
                        sent_b_index = sent_index

                        if sent_a_index == sent_b_index - 1:
                            da_example_neighbor_sent_pair_order_labels.append(0)    # b is next sentence of a and in the same topic
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(1)    # b is not next sentence of a but in the same topic
                elif tssp_ablation == "wo_intra_topic":
                    # wo_intra_topic means there just exists two class, where 0 means b and a are in the same topic, 1 means b and a are not in the same topic
                    if j == 0:
                        da_example_neighbor_sent_pair_order_labels.append(1)    # b is start sentence of new topic
                    else:
                        da_example_neighbor_sent_pair_order_labels.append(0)    # b and a are in the same topic
                elif tssp_ablation == "wo_inter_topic":
                    # wo_inter_topic means there just exists two class, where 0 means b is nsp of a, and 1 means b is not nsp of a. just like bert.
                    # specifically, nsp can be established only when 
                    # b and a are in the same topic, and b is nsp of a or
                    # b and a are not in the same topic, and the two topic are continus in original doc, and a is eot and b is eot.
                    if j == 0:
                        if i == 0:
                            da_example_neighbor_sent_pair_order_labels.append(1)        # first sentence of doc
                        else:
                            if topic_orders[i-1] == -1 or topic_orders[i-1] + 1 != topic_orders[i]:     # topic_orders[i-1] + 1 != topic_orders[i] is True when topic_orders[i] == -1
                                da_example_neighbor_sent_pair_order_labels.append(1)        # left or current topic is replaced by other doc or shuffled, so end sentence of left topic and current sentence are not nsp
                            else:
                                if sent_index == 0:
                                    da_example_neighbor_sent_pair_order_labels.append(0)        # it's also nsp although a and b are not in the same topic. since end sentence of left topic are not changed. 
                                else:
                                    da_example_neighbor_sent_pair_order_labels.append(1)        # sentences are shuffled and original start sentence are shuffled to other position
                    else:
                        sent_a_index = topic_sent_indices[j - 1]
                        sent_b_index = sent_index
                        if sent_a_index == sent_b_index - 1:
                            da_example_neighbor_sent_pair_order_labels.append(0)    # b is next sentence of a and in the same topic
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(1)
                elif tssp_ablation == "sso":
                    # sso is for implementing struct-bert. where 0 means Next Sent Prediction; 1 means Prev. Sent Prediction; 2 means Random Sent Prediction.
                    # NSP follow nsp in wo_inter_topic
                    # Prev. Sent Prediction
                    if j == 0:
                        if i == 0:
                            da_example_neighbor_sent_pair_order_labels.append(2)        # first sentence of doc don't to contribute to sop loss
                        else:
                            if topic_orders[i-1] == -1 or topic_orders[i-1] + 1 != topic_orders[i]:     # topic_orders[i-1] + 1 != topic_orders[i] is True when topic_orders[i] == -1
                                da_example_neighbor_sent_pair_order_labels.append(2)        # left or current topic is replaced by other doc or shuffled, so end sentence of left topic and current sentence are random relation
                            else:
                                if sent_index == 0:
                                    da_example_neighbor_sent_pair_order_labels.append(0)        # it's also nsp although a and b are not in the same topic. since end sentence of left topic are not changed. 
                                else:
                                    da_example_neighbor_sent_pair_order_labels.append(2)        # sentences are shuffled and original start sentence are shuffled to other position
                    else:
                        sent_a_index = topic_sent_indices[j - 1]
                        sent_b_index = sent_index
                        if sent_a_index == sent_b_index - 1:
                            da_example_neighbor_sent_pair_order_labels.append(0)    # b is next sentence of a and in the same topic
                        elif sent_a_index == sent_b_index + 1:
                            da_example_neighbor_sent_pair_order_labels.append(1)    # b is previous sentence of a. maybe this class label has little samples
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(2)
                elif tssp_ablation == "sso_and_intra_topic":
                    # where 0 means Next Sent Prediction; 1 means Prev. Sent Prediction; 2 means Random Sent Prediction
                    # NSP follow nsp in wo_inter_topic
                    # Prev. Sent Prediction
                    if j == 0:
                        da_example_neighbor_sent_pair_order_labels.append(2)    
                    else:
                        sent_a_index = topic_sent_indices[j - 1]
                        sent_b_index = sent_index
                        if sent_a_index == sent_b_index - 1:
                            da_example_neighbor_sent_pair_order_labels.append(0)    # b is next sentence of a and in the same topic
                        elif sent_a_index == sent_b_index + 1:
                            da_example_neighbor_sent_pair_order_labels.append(1)    # b is previous sentence of a. maybe this class label has little samples
                        else:
                            da_example_neighbor_sent_pair_order_labels.append(2)
                else:
                    raise ValueError("not recognized tssp_ablation %s" % tssp_ablation)
                
                start_token_index, end_token_index = example_sent_index_to_start_end_token_index[sent_index]
                sent_b_token_cnt = end_token_index - start_token_index + 1
                da_example_neighbor_sent_pair_order_labels += [-100 for _ in range(sent_b_token_cnt - 1)]
            
            da_example_sent_label_ids += new_topic_sent_label_ids
        return da_example_input_ids, da_example_sentences, da_example_sent_label_ids, da_example_neighbor_sent_pair_order_labels, replaced_flag
        
    def get_example_sent_index_to_start_end_token_index(example_input_ids, example_sentences):
        # record sentence index -> start and end index of sentence input_ids in example input_ids
        sent_start_token_index = []
        for token_index in range(len(example_input_ids)):
            if example_input_ids[token_index] in target_specical_ids:       # current token is bos
                sent_start_token_index.append(token_index)
        sent_end_token_index = [sent_start_token_index[_] - 1 for _ in range(1, len(sent_start_token_index))] + [len(example_input_ids) - 1]
        assert len(sent_start_token_index) == len(example_sentences)
        
        example_sent_index_to_start_end_token_index = {}
        for i in range(len(example_sentences)):
            example_sent_index_to_start_end_token_index[i] = (sent_start_token_index[i], sent_end_token_index[i])       # (start, end) is [] not [)
        
        return example_sent_index_to_start_end_token_index

    def prepare_augmented_data(sentences, labels, tokenized_examples):
        # data augmentation
        num_examples = len(sentences)
        all_sent_index_to_start_end_token_index = []    # record sentence index -> start and end index of sentence input_ids in example input_ids. each value is a dict of each example
        all_sent_index_to_sent_text = []
        all_sent_label = []
        all_topic_index_to_start_end_sent_index = []    # record topic index -> start and end index of sentence in example sentences
        for example_index in range(num_examples):
            example_sentences = sentences[example_index]
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_sent_labels = [label_to_id[_] if _ in label_to_id else -100 for _ in labels[example_index]]
            all_sent_label.append(example_sent_labels)

            example_sent_index_to_start_end_token_index = get_example_sent_index_to_start_end_token_index(example_input_ids, example_sentences)
            all_sent_index_to_start_end_token_index.append(example_sent_index_to_start_end_token_index)

            example_sent_index_to_sent_text = {}
            for i, sent_text in enumerate(example_sentences):
                example_sent_index_to_sent_text[i] = sent_text
            all_sent_index_to_sent_text.append(example_sent_index_to_sent_text)

            # record topic index -> start and end index of sentence in example sentences
            example_topic_index_to_start_end_sent_index = {}
            topic_end_sent_indices = [sent_index for sent_index, v in enumerate(example_sent_labels) if v == label_to_id["B-EOP"]]
            topic_start_sent_indices = [0] + [topic_end_sent_indices[i] + 1 for i in range(len(topic_end_sent_indices) - 1)]
            for i, (t_start_s_id, t_end_s_id) in enumerate(zip(topic_start_sent_indices, topic_end_sent_indices)):
                example_topic_index_to_start_end_sent_index[i] = (t_start_s_id, t_end_s_id)     # (start, end) is [] not [)
            all_topic_index_to_start_end_sent_index.append(example_topic_index_to_start_end_sent_index)
        
        da_input_ids = []   # we shuffle sentences in input_ids aspect rather than sentence text aspect. because number of tokens of replaced sentence may be different from origin sentence
        da_sentences = []
        da_sent_label_ids = []
        da_token_seq_labels = []
        da_neighbor_sent_pair_order_labels = []     # length equals to token sequence length. and if ith token is bos then the corresponding value is pair_order_label for the sentence
        da_example_replaced_flags = []              # annotate whether current example has some sentences from other examples
        for example_index in range(num_examples):
            example_sentences = sentences[example_index]
            example_sent_labels = [label_to_id[_] if _ in label_to_id else -100 for _ in labels[example_index]]
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_sent_index_to_start_end_token_index = all_sent_index_to_start_end_token_index[example_index]

            da_example_input_ids = []
            da_example_sentences = []
            da_example_sent_label_ids = []
            da_example_neighbor_sent_pair_order_labels = []     # length is equal to token sequence. and bos token has sent_pair_order_label, other tokens has -100.
            '''
            for neighbor sentence pair a and b in da example, the order labels are in one of 4 class according to origin example.
            0 means b is next sentence of a and in the same topic.
            1 means b is not next sentence of a but in the same topic.
            2 means b is start sentence of new topic which means a -> b is topic convertion.
            '''
            
            topic_end_sent_indices = [sent_index for sent_index, v in enumerate(example_sent_labels) if v == label_to_id["B-EOP"]]
            topic_start_sent_indices = [0] + [topic_end_sent_indices[i] + 1 for i in range(len(topic_end_sent_indices) - 1)]
            
            # shuffle and replace at probability topics in doc; then shuffle paragraphs in topic
            da_example_input_ids, da_example_sentences, \
            da_example_sent_label_ids, da_example_neighbor_sent_pair_order_labels, \
            replaced_flag, topic_orders = shuffle_and_replace_doc_topics(
                                tokenized_examples,
                                num_examples,
                                example_index,
                                example_sentences,
                                example_sent_labels,
                                example_sent_index_to_start_end_token_index,
                                example_input_ids,
                                topic_start_sent_indices, 
                                topic_end_sent_indices,
                                all_topic_index_to_start_end_sent_index, 
                                all_sent_index_to_sent_text,
                                all_sent_label,
                                all_sent_index_to_start_end_token_index,
                                )
            
            da_example_sent_index_to_start_end_token_index = example_sent_index_to_start_end_token_index = get_example_sent_index_to_start_end_token_index(da_example_input_ids, da_example_sentences)
            da_topic_end_sent_indices = [sent_index for sent_index, v in enumerate(da_example_sent_label_ids) if v == label_to_id["B-EOP"]]
            da_topic_start_sent_indices = [0] + [da_topic_end_sent_indices[i] + 1 for i in range(len(da_topic_end_sent_indices) - 1)]

            da_example_input_ids, da_example_sentences, \
            da_example_sent_label_ids, da_example_neighbor_sent_pair_order_labels, \
            _ = shuffle_topic_sents(
                da_example_input_ids,
                da_example_sentences,
                da_example_sent_label_ids,
                da_example_sent_index_to_start_end_token_index,
                da_topic_start_sent_indices,
                da_topic_end_sent_indices,
                tssp_ablation=config.tssp_ablation,
                topic_orders=topic_orders,
            )
            da_example_replaced_flags.append(replaced_flag)
            
            da_input_ids.append(da_example_input_ids)
            da_sentences.append(da_example_sentences)
            da_sent_label_ids.append(da_example_sent_label_ids)
            da_neighbor_sent_pair_order_labels.append(da_example_neighbor_sent_pair_order_labels)    

        # get da_token_seq_labels
        for example_index in range(num_examples):
            da_example_input_ids = da_input_ids[example_index]
            da_example_sent_label_ids = da_sent_label_ids[example_index]
            da_example_token_seq_labels = []
            cur_sent_index = -1
            for token_index in range(len(da_example_input_ids)):
                if da_example_input_ids[token_index] in target_specical_ids:       # current token is bos
                    cur_sent_index += 1
                    da_example_token_seq_labels.append(da_example_sent_label_ids[cur_sent_index])
                else:
                    da_example_token_seq_labels.append(-100)
            da_token_seq_labels.append(da_example_token_seq_labels)

        return da_input_ids, da_sentences, da_sent_label_ids, da_token_seq_labels, da_neighbor_sent_pair_order_labels, da_example_replaced_flags

    # Tokenize all texts and align the labels with them.
    def prepare_features_with_dynamic_num_sentence(examples):

        labels = examples[label_column_name]      # labels
        contexts = examples[context_column_name]        # sentences
        example_ids = examples[example_id_column_name]
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

        da_input_ids, da_sentences, da_sent_label_ids, da_token_seq_labels, da_neighbor_sent_pair_order_labels, da_example_replaced_flags = prepare_augmented_data(sentences, labels, tokenized_examples)

        # slide-window to convert example to samples
        new_example_ids = []
        new_sentences = []
        new_token_seq_labels = []
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []

        new_sent_level_labels = []
        new_extract_eop_segment_ids = []    # used for cl module
        new_eop_index_for_aggregate_batch_eop_features = []     # used for cl module. [0, 1, 2, 3, ..., k, 0, 0, 0] where k is the number of eop, first 0 is for cls, and other 0 is for padding
        new_sent_pair_orders = []           # used for tssp module
        new_sent_token_mask = []            # extracting bos token and corresponding topic_segment_ids. format is like [-100, 1, -100, .., 0, -100, ..., 1, -100, ...] where 1 or 0 is bos token and 0 means end sentence of topic

        for example_index in range(num_examples):
            example_id = example_ids[example_index]
            example_sentences = [str(i) + "-" + s for i, s in enumerate(contexts[example_index])]
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_token_seq_labels = tokenized_examples["token_seq_labels"][example_index]
            example_token_type_ids = tokenized_examples["token_type_ids"][example_index]
            example_attention_mask = tokenized_examples["attention_mask"][example_index]

            da_example_id = example_ids[example_index]
            da_example_sentences = da_sentences[example_index]
            da_example_input_ids = da_input_ids[example_index]
            da_example_token_seq_labels = da_token_seq_labels[example_index]
            da_example_token_type_ids = tokenized_examples["token_type_ids"][example_index]
            da_example_attention_mask = tokenized_examples["attention_mask"][example_index]

            replaced_flag = da_example_replaced_flags[example_index]
            
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
                    da_sample_input_ids = [tokenizer.cls_token_id] + da_example_input_ids[token_left_index:token_right_index]
                    da_sample_input_ids = da_sample_input_ids[:max_seq_length]

                    sample_token_seq_labels = [-100] + example_token_seq_labels[token_left_index:token_right_index]
                    sample_token_seq_labels = sample_token_seq_labels[:max_seq_length]
                    da_sample_token_seq_labels = [-100] + da_example_token_seq_labels[token_left_index:token_right_index]
                    da_sample_token_seq_labels = da_sample_token_seq_labels[:max_seq_length]

                    sample_token_type_ids = [0] + example_token_type_ids[token_left_index:token_right_index]
                    sample_token_type_ids = sample_token_type_ids[:max_seq_length]
                    da_sample_token_type_ids = [0] * len(da_sample_input_ids)

                    sample_attention_mask = [1] + example_attention_mask[token_left_index:token_right_index]
                    sample_attention_mask = sample_attention_mask[:max_seq_length]
                    da_sample_attention_mask = [1] * len(da_sample_input_ids)

                    pair_orders = [-100] + da_neighbor_sent_pair_order_labels[example_index][token_left_index:token_right_index]
                    pair_orders = pair_orders[:max_seq_length]

                    if sent_right_index - 1 == sent_left_index:
                        last_bos_index_in_sample = example_sent_bos_to_token_index[sent_left_index] - token_left_index + 1
                        sample_token_seq_labels[last_bos_index_in_sample] = -100
                        token_left_index = token_right_index
                    else:
                        last_bos_index_in_sample = example_sent_bos_to_token_index[sent_i] - token_left_index + 1
                        sample_token_seq_labels[last_bos_index_in_sample] = -100
                        token_left_index = accumulate_length[sent_i - 1] + 1      # neighboring windows share one sentence
                    
                    sample_sentences = example_sentences[sent_left_index:sent_right_index]
                    da_sample_sentences = da_example_sentences[sent_left_index:sent_right_index]

                    if sent_right_index - 1 == sent_left_index or token_right_index == example_total_num_tokens:
                        sent_left_index = sent_right_index
                        sent_i += 1
                    else:
                        sent_left_index = sent_right_index - 1
                    
                    actual_sample_token_length = len(sample_input_ids)
                    while len(sample_input_ids) < max_seq_length:
                        sample_input_ids.append(tokenizer.pad_token_id)
                        sample_token_seq_labels.append(-100)
                        sample_token_type_ids.append(0)
                        sample_attention_mask.append(0)
                    
                    while len(da_sample_input_ids) < max_seq_length:
                        da_sample_input_ids.append(tokenizer.pad_token_id)
                        da_sample_token_seq_labels.append(-100)
                        da_sample_token_type_ids.append(0)
                        da_sample_attention_mask.append(0)
                        pair_orders.append(-100)
                    
                    anchor_sample_sent_token_mask = get_sample_sent_token_mask(sample_input_ids, sample_token_seq_labels)
                    da_sample_sent_token_mask = get_sample_sent_token_mask(da_sample_input_ids, da_sample_token_seq_labels)

                    number_sent_1 = len([_ for _ in pair_orders if _ != -100])
                    number_sent_2 = len([_ for _ in da_sample_sent_token_mask if _ != -100])
                    assert number_sent_1 == number_sent_2
                    new_sent_token_mask.append([anchor_sample_sent_token_mask, da_sample_sent_token_mask])
                    new_sent_pair_orders.append([pair_orders, pair_orders])

                    new_example_ids.append([example_id, da_example_id])
                    new_sentences.append([sample_sentences, da_sample_sentences])
                    new_input_ids.append([sample_input_ids, da_sample_input_ids])
                    new_token_seq_labels.append([sample_token_seq_labels, da_sample_token_seq_labels])
                    new_token_type_ids.append([sample_token_type_ids, da_sample_token_type_ids])
                    new_attention_mask.append([sample_attention_mask, da_sample_attention_mask])

                    # sent_level_labels
                    sample_sent_level_labels = [-100]
                    for i in range(1, len(sample_input_ids)):
                        if sample_input_ids[i] in target_specical_ids:
                            sample_sent_level_labels.append(sample_token_seq_labels[i])
                    if len(sample_sent_level_labels) != max_seq_length:
                        # padding to max_seq_length due the dynamic sentence number
                        sample_sent_level_labels += [-100] * (max_seq_length - len(sample_sent_level_labels))
                    
                    da_sample_sent_level_labels = [-100]
                    for i in range(1, len(da_sample_input_ids)):
                        if da_sample_input_ids[i] in target_specical_ids:
                            da_sample_sent_level_labels.append(da_sample_token_seq_labels[i])
                    if len(da_sample_sent_level_labels) != max_seq_length:
                        da_sample_sent_level_labels += [-100] * (max_seq_length - len(da_sample_sent_level_labels))
                    new_sent_level_labels.append([sample_sent_level_labels, da_sample_sent_level_labels])

                    anchor_extract_eop_segment_ids = get_extract_eop_segment_ids(sample_input_ids, sample_token_seq_labels)
                    da_extrac_eop_segment_ids = get_extract_eop_segment_ids(da_sample_input_ids, da_sample_token_seq_labels)
                    new_extract_eop_segment_ids.append([anchor_extract_eop_segment_ids, da_extrac_eop_segment_ids])
                    
                    anchor_eop_cnt = len([v for v in sample_token_seq_labels if v != -100])
                    anchor_eop_index_for_aggregate_batch_eop_features = list(range(anchor_eop_cnt + 1)) + [0] * (max_seq_length - anchor_eop_cnt - 1)
                    da_eop_cnt = len([v for v in da_sample_token_seq_labels if v != -100])
                    da_eop_index_for_aggregate_batch_eop_features = list(range(da_eop_cnt + 1)) + [0] * (max_seq_length - da_eop_cnt - 1)
                    new_eop_index_for_aggregate_batch_eop_features.append([anchor_eop_index_for_aggregate_batch_eop_features, da_eop_index_for_aggregate_batch_eop_features])
                else:
                    sent_i += 1

        output_samples = {
            "example_id": new_example_ids,
            "labels": new_token_seq_labels,
            "sentences": new_sentences,
            "input_ids": new_input_ids,
            "token_type_ids": new_token_type_ids,
            "attention_mask": new_attention_mask,
            "sent_level_labels": new_sent_level_labels,
            "extract_eop_segment_ids": new_extract_eop_segment_ids,
            "eop_index_for_aggregate_batch_eop_features": new_eop_index_for_aggregate_batch_eop_features,
            "sent_pair_orders": new_sent_pair_orders,
            "sent_token_mask": new_sent_token_mask,
        }

        return output_samples

    train_preprocess_fn = prepare_features_with_dynamic_num_sentence
    valid_preprocess_fn = prepare_features_with_dynamic_num_sentence
    test_preprocess_fn = prepare_features_with_dynamic_num_sentence

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                train_preprocess_fn,
                batched=True,
                batch_size=10000,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, dataset_name, max_seq_length, "train"),
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
        training_args.logging_steps = eval_steps
        training_args.eval_steps = eval_steps
        training_args.save_steps = eval_steps
        print("eval_steps: ", eval_steps)

        for i in range(3):
            print("train example %d: " % i)
            print(train_dataset[i])

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                valid_preprocess_fn,
                batched=True,
                batch_size=10000,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, dataset_name, max_seq_length, "dev"),
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

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
                batch_size=10000,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                new_fingerprint="{}_{}_{}_{}".format(model_name, dataset_name, max_seq_length, "test",),
                desc="Running tokenizer on prediction dataset",
            )
    
    data_collator = default_data_collator
    
    # Metrics
    metric = load_metric(data_args.metric_name)
    def compute_metrics(p):
        all_predictions_logits, all_labels = p
        predictions_logits, eop_pair_cos_sim = all_predictions_logits
        labels, sent_level_labels = all_labels

        anchor_labels, da_labels = [v[0] for v in labels], [v[1] for v in labels]

        anchor_predictions_logits, da_predictions_logits = [v[0] for v in predictions_logits], [v[1] for v in predictions_logits]
        
        true_labels = [[label_list[l] for l in label if l != -100] for label in anchor_labels]
        da_true_labels = [[label_list[l] for l in label if l != -100] for label in da_labels]

        if model_args.ts_score_predictor == "lt":
            # predictions_logits and labels is token-level
            predictions = np.argmax(anchor_predictions_logits, axis=2)     # 0 will be converted to B-EOP by label_list, so 0 means segmentation point
            # Remove ignored index (special tokens)
            # label_list:  ['B-EOP', 'O']
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, anchor_labels)
            ]

            da_predictions = np.argmax(da_predictions_logits, axis=2)
            da_true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(da_predictions, da_labels)]
        elif model_args.ts_score_predictor == "cos":
            # predictions_logits is eop level. but labels is token-level
            predictions = (predictions_logits > 0.5).astype(np.int32)          # if sigmoid of cosine similarity > 0.5, then it's similarity and prediction is 1 which is not segmentation point
            predictions = [prediction[:len(label)] for prediction, label in zip(predictions, true_labels)]
            true_predictions = [[label_list[p] for p in prediction] for prediction in predictions]
        else:
            raise ValueError("not supported ts_score_predictor %s" % model_args.ts_score_predictor)
        
        results = metric.compute(predictions=true_predictions, references=true_labels)

        da_results = metric.compute(predictions=da_true_predictions, references=da_true_labels)
        da_results_with_prefix = {}
        for k, v in da_results.items():
            da_results_with_prefix["da_" + k] = v
        results.update(da_results_with_prefix)
        
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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
    if training_args.do_predict:
        metric_key_prefix = "predict"
        logger.info("*** Predict test set ***")
        sentences = predict_dataset["sentences"]
        example_ids = predict_dataset["example_id"]
        num_examples = len(predict_examples[context_column_name])
        num_samples = len(predict_dataset["sentences"])
        example_ids = [v[0] for v in example_ids]    # just get anchor_example_ids
        print("num predict samples: ", num_samples)
        print("num predict examples: ", num_examples)

        start_time = time.time()
        all_predict_logits, all_labels, metrics = trainer.predict(predict_dataset, metric_key_prefix=metric_key_prefix)
        predict_logits, eop_pair_cos_sim = all_predict_logits
        labels, sent_level_labels = all_labels
        end_time = time.time()
        print("predict_time(s): ", end_time - start_time)
        
        anchor_labels, da_labels = [v[0] for v in labels], [v[1] for v in labels]
        true_labels = [[label_list[l] for l in label if l != -100] for label in anchor_labels]
        true_int_labels = [[l for l in label if l != -100] for label in anchor_labels]
        anchor_predictions_logits, da_predictions_logits = [v[0] for v in predict_logits], [v[1] for v in predict_logits]
        if model_args.ts_score_predictor == "lt":
            predictions = np.argmax(anchor_predictions_logits, axis=2)
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, anchor_labels)
            ]
            true_predict_logits = [
                [p.tolist() for (p, l) in zip(p_logits, label) if l != -100]
                for p_logits, label in zip(anchor_predictions_logits, anchor_labels)
            ]
        elif model_args.ts_score_predictor == "cos":

            predictions = (predict_logits > 0.5).astype(np.int32)          # if sigmoid of cosine similarity > 0.5, then it's similar and prediction is 1 which is not segmentation point
            predictions = [prediction[:len(label)] for prediction, label in zip(predictions, true_int_labels)]
            true_predictions = [[label_list[p] for p in prediction] for prediction in predictions]
            true_predict_logits = [example_logits[:len(label)].tolist() for example_logits, label in zip(predict_logits, true_int_labels)]
        else:
            raise ValueError("not supported ts_score_predictor %s" % model_args.ts_score_predictor)
        
        if eop_pair_cos_sim is not None:
            true_eop_pair_cos_sim = [
                [v for v in cos_sim if v != -100]
                for cos_sim in eop_pair_cos_sim
            ]
        
        metric_file_name = "_".join([metric_key_prefix, custom_args.test_data_name, "max_seq%d" % data_args.max_seq_length, "ts_score_%s" % model_args.ts_score_predictor])
        trainer.log_metrics(metric_file_name, metrics)
        trainer.save_metrics(metric_file_name, metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, metric_file_name + ".txt")
        out = [{"sentences":[], "labels":[], "int_labels": [], "predictions":[], "predict_logits":[]} for _ in range(num_examples)]
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction, sentence_list, label, int_labels, example_id, predict_logits in zip(
                    true_predictions, sentences, true_labels, true_int_labels, example_ids, true_predict_logits):
                    out[example_id]["sentences"].extend(sentence_list)      # TODO overlap sentences
                    out[example_id]["labels"].extend(label)
                    out[example_id]["predictions"].extend(prediction)
                    out[example_id]["predict_logits"].extend(predict_logits)
                    out[example_id]["int_labels"].extend([int(v) for v in int_labels])

                if true_eop_pair_cos_sim is not None:
                    for i in range(num_examples):
                        out[i]["eop_pair_cos_sim"] = []
                    for example_id, eop_pair_cos_sim in zip(example_ids, true_eop_pair_cos_sim):
                        out[example_id]["eop_pair_cos_sim"].extend([float(v) for v in eop_pair_cos_sim])

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
            example_level_metric = metric.compute_metric_example_level(logits_not_empty, labels_not_empty, label_list, custom_args, data_args, ts_score_predictor=model_args.ts_score_predictor, mode=metric_key_prefix)
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


if __name__ == "__main__":
    main()