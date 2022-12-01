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


from datasets import Dataset
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Metrics
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import sys

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
from collections import defaultdict
from datasets import load_dataset, load_metric, ClassLabel, DatasetDict
from collections import deque
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
import torch


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
        metadata={"help": "sentence representation 获取方式，默认获取eos 表征，还只是segment mean(VALUE: mean) or segment max(VALUE: max)"},
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

    use_paragraph_segment: bool = field(
        default=False,
        metadata={
            "help": (
                "ponet模型专用参数，是否使用段落维度的segment id，默认是句子维度"
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


if __name__ == "__main__":

    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_id = model_args.model_name_or_path
    WORK_DIR = training_args.output_dir

    def data_parse_fn(all_examples, split, task):
        all_examples = all_examples["content"]
        print("task: {}; split: {}".format(task, split))
        label_map = {"1":"B-EOP", "0":"O", 1:"B-EOP", 0:"O"}
        out = []

        if task == "action_detection":

            for example_id, example in enumerate(all_examples):
                example = json.loads(example)

                sentences, action_sent_ids = example["sentences"], example["action_ids"]
                sentences = [_["s"] for _ in sentences]
                action_sent_ids = [_["id"] for _ in action_sent_ids]
                action_labels = [0] * len(sentences)
                for i in range(len(sentences)):
                    # segment id 从1开始
                    if i + 1 in action_sent_ids:
                        action_labels[i] = 1
                labels = action_labels
                for s, l in zip(sentences, labels):
                    out.append({
                        "example_id": example_id,
                        "sentence": s,
                        "label": l
                    })
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

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    print (train_dataset)

    max_epochs = int(training_args.num_train_epochs)
    lr = training_args.learning_rate
    batch_size = training_args.per_device_train_batch_size

    def cfg_modify_fn(cfg):
        cfg.task = Tasks.text_classification

        cfg.train.max_epochs = max_epochs
        cfg.train.optimizer.lr = lr
        cfg.train.dataloader = {
            "batch_size_per_gpu": batch_size,
            "workers_per_gpu": 1
        }

        cfg.evaluation.metrics = [Metrics.seq_cls_metric]
        cfg.train.lr_scheduler = {
            'type': 'LinearLR',
            'start_factor': 1.0,
            'end_factor': 0.0,
            'total_iters':
                int(len(train_dataset) / batch_size) * cfg.train.max_epochs,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.hooks[-1] = {
            'type': 'EvaluationHook',
            'by_epoch': True,
            'interval': 1
        }
        cfg['dataset'] = {
            'train': {
                'labels': ['否', '是', 'None'],
                'first_sequence': 'sentence',
                'label': 'label',
            }
        }
        return cfg

    # map float to index
    def map_labels(examples):
        map_dict = {0: "否", 1: "是"}
        examples['label'] = map_dict[int(examples['label'])]
        return examples

    train_dataset = train_dataset.map(map_labels)
    eval_dataset = eval_dataset.map(map_labels)

    if training_args.do_train:
        kwargs = dict(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=WORK_DIR,
            cfg_modify_fn=cfg_modify_fn)

        trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)

        print('===============================================================')
        print('pre-trained model loaded, training started:')
        print('===============================================================')

        trainer.train()

        print('===============================================================')
        print('train success.')
        print('===============================================================')

        for i in range(max_epochs):
            eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
            print(f'epoch {i} evaluation result:')
            print(eval_results)

        print('===============================================================')
        print('evaluate success')
        print('===============================================================')


# ---------------------------- Inference ---------------------------------
    if training_args.do_predict:
        output_list = []
        text_classification = pipeline(Tasks.text_classification, model=f'{WORK_DIR}/output')
        for sample in test_dataset:
            input_text = sample["sentence"]
            output = text_classification(input_text)
            scores = output["scores"]
            if scores[1] > scores[2]:
                label = output["labels"][1]
            else:
                label = output["labels"][2]
            output_list.append(input_text + "\t" + label.replace("是", "1").replace("否", "0"))
        with open(f'{WORK_DIR}/test_predict_result.txt', "w") as f:
            f.write("\n".join(output_list))

