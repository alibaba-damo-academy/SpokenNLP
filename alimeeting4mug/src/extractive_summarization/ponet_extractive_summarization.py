# coding=utf-8
# Copyright 2022 Alibaba.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import datasets
from datasets import load_dataset, load_metric, ClassLabel, DatasetDict
from collections import defaultdict
import transformers
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    Trainer,
    PretrainedConfig,
    BertTokenizerFast,
    BertConfig
)
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
import torch
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from utils.tokenizer import BasicTokenizer

bTokenizer = BasicTokenizer()
tokenize_func = bTokenizer.tokenize


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.11.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    sentence_pooler_type: str = field(
        default=None,
        metadata={"help": "sentence representation"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="paragraph_segment", metadata={"help": "The name of the task (ner, pos...)."})
    metric_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the metric to use (via the datasets library)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_sentence_length: int = field(
        default=30,
        metadata={
            "help": "The maximum context input sentence number. sampele longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    doc_sentence_stride: int = field(default=15)

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    def data_parse_fn(all_examples, split, task):
        all_examples = all_examples["content"]
        print("task: {}; split: {}".format(task, split))
        label_map = {"1": "B-EOP", "0": "O", 1:"B-EOP", 0:"O"}

        out = []
        if task == "doc_key_sentence_extraction":
            if split == "train":
                # 多人标注结果做聚合逻辑
                strategy = "union"
                # strategy = "single"
                # strategy = "pool"
                # strategy = "major_vote"

            else:
                # 多人标注结果不做聚合逻辑
                strategy = "single"
            print(split, strategy)
            example_id = -1
            label = "NONE"

            for example in all_examples:
                example = json.loads(example)
                if not example["topic_segment_ids"]:
                    example["topic_segment_ids"].append({"id":example["sentences"][-1]["id"]})
                if example["topic_segment_ids"][-1]["id"] != example["sentences"][-1]["id"]:
                    if example["topic_segment_ids"][-1]["id"] < example["sentences"][-1]["id"]:
                        example["topic_segment_ids"][-1]["id"] = example["sentences"][-1]["id"]

                paragraph_seg_key = "org_segment_id" if "org_segment_id" in example else "paragraph_segment_ids"
                sentences, topic_seg_ids, paragraph_seg_ids = example["sentences"], example["topic_segment_ids"], example[paragraph_seg_key]
                sentences = [_["s"] for _ in sentences]
                doc_key_sentence_labels = ["O"] * len(sentences)
                tmp = set()
                if "candidate" not in example:
                    example["candidate"] = []
                tmp.update([int(sent_id) for can in example["candidate"] for sent_id in can["key_sentence"]])
                doc_key_sentence_ids = sorted(list(tmp))
                for i in range(len(sentences)):
                    if i+1 in doc_key_sentence_ids:
                        doc_key_sentence_labels[i] = label_map[1]

                multi_ref_doc_key_sentence_labels = []
                if example["candidate"]:
                    for can in example["candidate"]:
                        multi_ref_doc_key_sentence_labels.append(["O"] * len(sentences))
                        for sent_id in can["key_sentence"]:
                            multi_ref_doc_key_sentence_labels[-1][int(sent_id)-1] = label_map[1]
                else:
                    ref_num = 3
                    multi_ref_doc_key_sentence_labels = [doc_key_sentence_labels] * ref_num

                if strategy == "union":
                    labels = doc_key_sentence_labels
                elif strategy == "single":
                    labels = multi_ref_doc_key_sentence_labels[0]
                elif strategy == "pool":
                    labels = multi_ref_doc_key_sentence_labels[0]
                    for other_labels in multi_ref_doc_key_sentence_labels[1:]:
                        example_id += 1
                        out.append({
                            "example_id": example_id,
                            "sentences": sentences,
                            "labels": other_labels,
                            "multi_labels": multi_ref_doc_key_sentence_labels,
                            "label": label
                        })
                elif strategy == "major_vote":
                    labels = []
                    for s_index in range(len(sentences)):
                        multi_l = [_[s_index] == label_map[1] for _ in multi_ref_doc_key_sentence_labels]
                        if sum(multi_l) > 1:
                            labels.append(label_map[1])
                        else:
                            labels.append(label_map[0])
                else:
                    raise NotImplementedError

                assert len(sentences) == len(labels)
                example_id += 1
                out.append({
                    "example_id": example_id,
                    "sentences":sentences,
                    "labels": labels,
                    "multi_labels":multi_ref_doc_key_sentence_labels,
                    "label": label
                })
        elif task == "topic_key_sentence_extraction":

            if split == "train":
                # 多人标注结果做聚合逻辑
                strategy = "union"
                # strategy = "single"
                # strategy = "pool"
                # strategy = "major_vote"
            else:
                # 多人标注结果不做聚合逻辑
                strategy = "single"

            topic_samples = []
            for example_id, example in enumerate(all_examples):
                example = json.loads(example)
                if not example["topic_segment_ids"]:
                    example["topic_segment_ids"].append({"id":example["sentences"][-1]["id"]})
                if example["topic_segment_ids"][-1]["id"] != example["sentences"][-1]["id"]:
                    if example["topic_segment_ids"][-1]["id"] < example["sentences"][-1]["id"]:
                        example["topic_segment_ids"][-1]["id"] = example["sentences"][-1]["id"]

                paragraph_seg_key = "org_segment_id" if "org_segment_id" in example else "paragraph_segment_ids"
                sentences, topic_seg_ids, paragraph_seg_ids = example["sentences"], example["topic_segment_ids"], \
                                                              example[paragraph_seg_key]
                sentences = [_["s"] for _ in sentences]
                topics = topic_seg_ids
                topic_left_index = 0
                for topic in topics:
                    topic_seg_id = int(topic["id"])
                    topic_key_sentence_labels = ["O"] * len(sentences)
                    tmp = set()
                    if "candidate" not in topic:
                        topic["candidate"] = []
                    tmp.update([int(sent_id) for can in topic["candidate"] for sent_id in can["key_sentence"]])
                    topic_key_sentence_ids = sorted(list(tmp))

                    for i in range(len(sentences)):
                        if i + 1 in topic_key_sentence_ids:
                            topic_key_sentence_labels[i] = label_map[1]

                    multi_ref_topic_key_sentence_labels = []
                    if topic["candidate"]:
                        for can in topic["candidate"]:
                            multi_ref_topic_key_sentence_labels.append(["O"] * len(sentences))
                            for sent_id in can["key_sentence"]:
                                multi_ref_topic_key_sentence_labels[-1][int(sent_id) - 1] = label_map[1]
                    else:
                        ref_num = 3
                        multi_ref_topic_key_sentence_labels = [topic_key_sentence_labels] * ref_num

                    if strategy == "union":
                        labels = topic_key_sentence_labels[topic_left_index:topic_seg_id]
                    elif strategy == "single":
                        labels = multi_ref_topic_key_sentence_labels[0][topic_left_index:topic_seg_id]
                    elif strategy == "pool":
                        labels = multi_ref_topic_key_sentence_labels[0][topic_left_index:topic_seg_id]
                        for other_labels in multi_ref_topic_key_sentence_labels[1:]:
                            topic_samples.append({
                                "sentences": sentences[topic_left_index:topic_seg_id],
                                "labels": other_labels[topic_left_index:topic_seg_id],
                                "label": "NONE",
                                "multi_labels": [_[topic_left_index:topic_seg_id] for _ in multi_ref_topic_key_sentence_labels]
                            })
                    elif strategy == "major_vote":
                        labels = []
                        for s_index in range(len(sentences)):
                            multi_l = [_[s_index] == label_map[1] for _ in multi_ref_topic_key_sentence_labels]
                            if sum(multi_l) > 1:
                                labels.append(label_map[1])
                            else:
                                labels.append(label_map[0])
                        labels = labels[topic_left_index:topic_seg_id]
                    else:
                        raise NotImplementedError

                    multi_ref_topic_key_sentence_labels = [_[topic_left_index:topic_seg_id] for _ in multi_ref_topic_key_sentence_labels]

                    topic_sentences = sentences[topic_left_index:topic_seg_id]
                    topic_left_index = topic_seg_id

                    topic_samples.append({
                        "sentences": topic_sentences,
                        "labels": labels,
                        "label": "NONE",
                        "multi_labels":multi_ref_topic_key_sentence_labels
                    })

            for example_id, topic in enumerate(topic_samples):
                topic["example_id"] = example_id
                out.append(topic)
        else:
            raise NotImplementedError

        out_dict = defaultdict(list)
        for sample in out:
            for key in sample.keys():
                out_dict[key].append(sample[key])
        # print(out_dict)
        return out_dict

    def alimeeting4mug_data_download():
        from modelscope.hub.api import HubApi
        from modelscope.msdatasets import MsDataset
        from modelscope.utils.constant import DownloadMode
        api = HubApi()
        sdk_token = ""  # 必填
        assert sdk_token, "从modelscope WEB端个人中心获取"
        api.login(sdk_token)  # online

        input_config_kwargs = {'delimiter': '\t'}
        data = MsDataset.load(
            'Alimeeting4MUG',
            namespace='modelscope',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            subset_name="default",
            **input_config_kwargs)

        return DatasetDict(
            {
                k: dataset.map(data_parse_fn, batched=True, batch_size=100, remove_columns=["idx", "content"], fn_kwargs={"split":k, "task":data_args.dataset_config_name})
                for k, dataset in data
            }
        )

    raw_datasets = alimeeting4mug_data_download()

    features = raw_datasets["train"].features
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
    label_column_name = "labels"
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        if data_args.task_name == "topic_segment":
            label_list = [_ for _ in label_list if _ != "-100"]
        label_to_id = {l: i for i, l in enumerate(label_list)}
    print("label_list: {}".format(" ".join(label_list)))
    num_labels = len(label_list)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if "PoNet".lower() in model_args.model_name_or_path.lower():
        model_type = "ponet"
        print("base model {}".format(model_type))
        from models.modeling_ponet import PoNetForTokenClassification
        model = PoNetForTokenClassification.from_pretrained(model_name_or_path=model_args.model_name_or_path,
                                                            task="token-classification-task", revision="v1.1.0")

        config = BertConfig.from_pretrained(
            model.model_dir,
            num_labels=num_labels,
            finetuning_task="ner",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            max_sentence_length=data_args.max_sentence_length,
            sentence_pooler_type=model_args.sentence_pooler_type
        )

        tokenizer = BertTokenizerFast.from_pretrained(
            model.model_dir,
            use_fast=True,
            use_auth_token=False,
        )
        config.gradient_checkpointing = True
        config.use_cache = False
    else:
        raise NotImplementedError

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    question_column_name = label_column_name
    context_column_name = "sentences"
    example_id_column_name = "example_id"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    num_added_toks = tokenizer.add_special_tokens({'eos_token':'[EOS]'})
    model.resize_token_embeddings(len(tokenizer))

    # extend position embeddings
    if model_type == "ponet" and data_args.max_seq_length > tokenizer.model_max_length:
        max_pos = data_args.max_seq_length
        config.max_position_embeddings = max_pos
        tokenizer.model_max_length = max_pos
        tokenizer.init_kwargs['model_max_length'] = tokenizer.model_max_length
        current_max_pos, embed_size = model.ponet.embeddings.position_embeddings.weight.shape
        assert max_pos > current_max_pos
        # allocate a larger position embedding matrix
        new_pos_embed = model.ponet.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        step = current_max_pos
        while k < max_pos - 1:
            new_pos_embed[k:(k + step)] = model.ponet.embeddings.position_embeddings.weight[:]
            k += step
        model.ponet.embeddings.position_embeddings.weight.data = new_pos_embed
        model.ponet.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    target_specical_ids = set()
    target_specical_ids.add(tokenizer.eos_token_id)
    print("target special id: ", target_specical_ids)
    print("target special id: ", tokenizer.decode(tokenizer.eos_token_id))

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

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def prepare_input_features(examples):

        questions = examples[question_column_name]
        contexts = examples[context_column_name]
        example_ids = examples[example_id_column_name]
        num_examples = len(questions)
        print("num examples: ", num_examples)

        sentences = []
        for example_index in range(num_examples):
            sentence_list = []
            for l, s in zip(questions[example_index], contexts[example_index]):
                add_suffix = ""
                if l in label_to_id :
                    add_suffix += "[SEP][EOS]"
                else:
                    add_suffix += "[EOS]"
                add_suffix = "[EOS]"
                sentence_list.append(s + add_suffix)
            sentences.append(sentence_list)

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
            print(examples[question_column_name])
            print("++++ERROR++++")
            return {}
        # print("num features", len(tokenized_examples))
        print("num tokenized_examples: ", len(tokenized_examples["input_ids"]))
        # print(tokenized_examples.keys())

        segment_ids = []
        token_seq_labels = []
        for example_index in range(num_examples):
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_labels = questions[example_index]
            example_labels = [label_to_id[_] if _ in label_to_id else -100 for _ in example_labels]
            # print("example_input_ids", example_input_ids)
            # print("example_labels", example_labels)
            example_token_labels = []
            segment_id = []
            cur_seg_id = 1
            for token_index in range(len(example_input_ids)):
                if example_input_ids[token_index] in target_specical_ids:
                    # print("append example label: ", example_labels[cur_seg_id - 1])
                    example_token_labels.append(example_labels[cur_seg_id - 1])
                    segment_id.append(cur_seg_id)
                    cur_seg_id += 1
                else:
                    example_token_labels.append(-100)
                    segment_id.append(cur_seg_id)
            segment_ids.append(segment_id)
            token_seq_labels.append(example_token_labels)
        tokenized_examples["segment_ids"] = segment_ids
        tokenized_examples["token_seq_labels"] = token_seq_labels

        new_segment_ids = []
        new_token_seq_labels = []
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []
        new_example_ids = []
        new_sentences = []
        for example_index in range(num_examples):
            example_input_ids = tokenized_examples["input_ids"][example_index]
            example_token_type_ids = tokenized_examples["token_type_ids"][example_index]
            example_attention_mask = tokenized_examples["attention_mask"][example_index]
            example_segment_ids = tokenized_examples["segment_ids"][example_index]
            example_token_seq_labels = tokenized_examples["token_seq_labels"][example_index]
            example_sentences = contexts[example_index]
            example_id = example_ids[example_index]
            example_total_num_sentences = len(questions[example_index])
            example_total_num_tokens = len(tokenized_examples["input_ids"][example_index])
            accumulate_length = [i for i, x in enumerate(tokenized_examples["input_ids"][example_index]) if x == tokenizer.eos_token_id]

            samples_boundary = []
            left_index = 0
            sent_left_index = 0
            sent_i = 0
            # for sent_i, length in enumerate(accumulate_length):
            while sent_i < len(accumulate_length):
                length = accumulate_length[sent_i]
                right_index = length + 1
                sent_right_index = sent_i + 1
                if right_index - left_index >= max_seq_length - 1 or right_index == example_total_num_tokens:
                    samples_boundary.append([left_index, right_index])

                    sample_input_ids = [tokenizer.cls_token_id] + example_input_ids[left_index:right_index]
                    sample_input_ids = sample_input_ids[:max_seq_length]

                    sample_token_type_ids = [0] + example_token_type_ids[left_index:right_index]
                    sample_token_type_ids = sample_token_type_ids[:max_seq_length]

                    sample_attention_mask = [1] + example_attention_mask[left_index:right_index]
                    sample_attention_mask = sample_attention_mask[:max_seq_length]

                    sample_segment_ids = [0] + example_segment_ids[left_index:right_index]
                    sample_segment_ids = sample_segment_ids[:max_seq_length]

                    sample_token_seq_labels = [-100] + example_token_seq_labels[left_index:right_index]
                    sample_token_seq_labels = sample_token_seq_labels[:max_seq_length]

                    if sent_right_index-1 == sent_left_index:
                        left_index = right_index
                        sample_input_ids[-1] = tokenizer.eos_token_id
                        sample_token_seq_labels[-1] = -100
                    else:
                        left_index = accumulate_length[sent_i - 1] + 1
                        if sample_token_seq_labels[-1] != -100:
                            sample_token_seq_labels[-1] = -100

                    if sent_right_index-1 == sent_left_index or right_index == example_total_num_tokens:
                        # print("sample_sentences 1", sent_left_index, sent_right_index, right_index, example_total_num_tokens)
                        sample_sentences = example_sentences[sent_left_index:sent_right_index]
                        sent_left_index = sent_right_index
                        sent_i += 1
                    else:
                        # print("sample_sentences 2", sent_left_index, sent_right_index-1)
                        sample_sentences = example_sentences[sent_left_index:sent_right_index-1]
                        sent_left_index = sent_right_index - 1

                    # print("len(sample_input_ids); ", len(sample_input_ids))
                    while len(sample_input_ids) < max_seq_length:
                        sample_input_ids.append(tokenizer.pad_token_id)
                        sample_token_type_ids.append(0)
                        sample_attention_mask.append(0)
                        sample_segment_ids.append(example_total_num_sentences + 1)
                        sample_token_seq_labels.append(-100)

                    new_input_ids.append(sample_input_ids)
                    new_token_type_ids.append(sample_token_type_ids)
                    new_attention_mask.append(sample_attention_mask)
                    new_segment_ids.append(sample_segment_ids)
                    new_token_seq_labels.append(sample_token_seq_labels)
                    new_example_ids.append(example_id)
                    new_sentences.append(sample_sentences)
                else:
                    sent_i += 1
                    continue

        output_samples = {}

        output_samples["input_ids"] = new_input_ids
        output_samples["token_type_ids"] = new_token_type_ids
        output_samples["attention_mask"] = new_attention_mask

        output_samples["segment_ids"] = new_segment_ids
        output_samples["example_id"] = new_example_ids
        output_samples["labels"] = new_token_seq_labels
        output_samples["sentences"] = new_sentences

        return output_samples

    train_preprocess_fn = prepare_input_features
    valid_preprocess_fn = prepare_input_features
    test_preprocess_fn = prepare_input_features

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]


        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                train_preprocess_fn,
                batched=True,
                batch_size=10000,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(min(eval_examples.shape[0], data_args.max_eval_samples)))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                valid_preprocess_fn,
                batched=True,
                batch_size=10000,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(min(eval_dataset.shape[0], data_args.max_eval_samples)))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                test_preprocess_fn,
                batched=True,
                batch_size=10000,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    metric = load_metric(data_args.metric_name)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        # do example level metric computation
        if len(predictions) == len(predict_dataset["sentences"]):
            num_examples = len(predict_examples[context_column_name])
            multi_labels = predict_examples["multi_labels"]
            num_samples = len(predict_dataset["sentences"])

            sentences = predict_dataset["sentences"]
            example_ids = predict_dataset["example_id"]
        elif len(predictions) == len(eval_dataset["sentences"]):
            num_examples = len(eval_examples[context_column_name])
            multi_labels = eval_examples["multi_labels"]
            num_samples = len(eval_dataset["sentences"])

            sentences = eval_dataset["sentences"]
            example_ids = eval_dataset["example_id"]
        else:
            raise Exception("not implementation.")

        out = []
        for i in range(num_examples):
            out.append({"sentences":[], "labels":[], "predictions":[]})

        rouge_predictions = []
        rouge_references = []

        for prediction, sentence_list, label, example_id in zip(true_predictions, sentences, true_labels, example_ids):
            if len(label) < len(sentence_list):
                label.append("O")
                prediction.append("O")
            assert len(sentence_list) == len(prediction), "{} {}".format(len(sentence_list), len(prediction))
            assert len(sentence_list) == len(label), "{} {}".format(len(sentence_list), len(label))
            out[example_id]["sentences"].extend(sentence_list)
            out[example_id]["labels"].extend(label)
            out[example_id]["predictions"].extend(prediction)

        for sample_index, sample in enumerate(out):
            doc = sample["sentences"]
            label = sample["labels"]
            prediction = sample["predictions"]
            pred_sent = [" ".join(tokenize_func(sent)) for sent, l, p in zip(doc, label, prediction) if
                         p == "B-EOP"]
            label_sent = [" ".join(tokenize_func(sent)) for sent, l, p in zip(doc, label, prediction) if
                          l == "B-EOP"]
            pred_sent = pred_sent if pred_sent else [" "]
            label_sent = label_sent if label_sent else [" "]
            sample["pred_sentences"] = pred_sent
            sample["label_sentences"] = label_sent
            rouge_predictions.append(pred_sent)
            rouge_references.append(label_sent)

        rouge_res = metric.rouge_compute(rouge_predictions, rouge_references)

        assert len(multi_labels) == len(out)
        rouge_multi_references = []
        for example_multi_ref, doc in zip(multi_labels, out):
            multi_ref_sentences = []
            doc = doc["sentences"]
            for ref in example_multi_ref:
                assert len(ref) == len(doc)
                ref_sent = [" ".join(tokenize_func(sent)) for sent, l in zip(doc, ref) if
                l == "B-EOP"]
                if not ref_sent:
                    ref_sent = [" "]
                multi_ref_sentences.append(ref_sent)
            rouge_multi_references.append(multi_ref_sentences)

        multi_rouge_scores_average = []
        multi_rouge_scores_max = []
        for pred, multi_ref in zip(rouge_predictions, rouge_multi_references):
            multi_rouge = [metric.rouge_compute([pred], [_]) for _ in multi_ref]
            # print(multi_rouge)
            max_score = max(multi_rouge, key=lambda x: x["rouge-l_f"])
            multi_rouge_scores_max.append(max_score)

            ave_score = {}
            for key in max_score.keys():
                ave = np.mean([_[key] for _ in multi_rouge])
                ave_score[key] = ave
            multi_rouge_scores_average.append(ave_score)

        multi_ref_ave_rouge = {}
        multi_ref_max_rouge = {}
        for key in max_score.keys():
            ave = np.mean([_[key] for _ in multi_rouge_scores_average])
            multi_ref_ave_rouge["multi-ref-average_{}".format(key)] = ave

            ave = np.mean([_[key] for _ in multi_rouge_scores_max])
            multi_ref_max_rouge["multi-ref-max_{}".format(key)] = ave

        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
        else:
            final_results = {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        final_results.update(rouge_res)
        final_results.update(multi_ref_ave_rouge)
        final_results.update(multi_ref_max_rouge)
        return final_results

    # Initialize our Trainer
    from models.trainer import MyTrainer
    trainer = MyTrainer(
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
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
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

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # print(predict_dataset.column_names)
        # print(predict_dataset["sentences"][:10])
        num_examples = len(predict_examples[context_column_name])
        num_samples = len(predict_dataset["sentences"])

        sentences = predict_dataset["sentences"]
        example_ids = predict_dataset["example_id"]

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)
        assert len(sentences) == len(predictions), "sample {}  infer_sample {} prediction {}".format(num_samples, len(sentences), len(predictions))
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        metrics["predict_samples"] = num_samples

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        out = []
        for i in range(num_examples):
            out.append({"sentences":[], "labels":[], "predictions":[]})
        if trainer.is_world_process_zero():
            rouge_predictions = []
            rouge_references = []

            with open(output_predictions_file, "w") as writer:
                for prediction, sentence_list, label, example_id in zip(true_predictions, sentences, true_labels, example_ids):
                    if len(label) < len(sentence_list):
                        label.append("O")
                        prediction.append("O")
                    assert len(sentence_list) == len(prediction), "{} {}".format(len(sentence_list), len(prediction))
                    assert len(sentence_list) == len(label), "{} {}".format(len(sentence_list), len(label))
                    out[example_id]["sentences"].extend(sentence_list)
                    out[example_id]["labels"].extend(label)
                    out[example_id]["predictions"].extend(prediction)

                for sample_index, sample in enumerate(out):
                    doc = sample["sentences"]
                    label = sample["labels"]
                    prediction = sample["predictions"]
                    pred_sent = [" ".join(tokenize_func(sent)) for sent, l, p in zip(doc, label, prediction) if
                                 p == "B-EOP"]
                    label_sent = [" ".join(tokenize_func(sent)) for sent, l, p in zip(doc, label, prediction) if
                                  l == "B-EOP"]
                    pred_sent = pred_sent if pred_sent else [" "]
                    label_sent = label_sent if label_sent else [" "]
                    sample["pred_sentences"] = pred_sent
                    sample["label_sentences"] = label_sent
                    rouge_predictions.append(pred_sent)
                    rouge_references.append(label_sent)

                writer.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in out])


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
