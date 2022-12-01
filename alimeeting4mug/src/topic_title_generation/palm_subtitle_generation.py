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

import os

from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics, Trainers
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.metrics import TextGenerationMetric
from datasets import Dataset, DatasetDict
from collections import defaultdict
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import datasets
from datasets import load_dataset, load_metric, ClassLabel
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

        if task == "topic_title_generation":
            if split == "train":
                # 多人标注结果做聚合逻辑
                # strategy = "single"
                strategy = "pool"
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
                    if "candidate" not in topic:
                        num_ref = 3
                        topic_titles = [""] * 3
                    else:
                        topic_titles = [candidate["title"] for candidate in topic["candidate"]]

                    if strategy == "single":
                        labels = topic_titles[0]
                        topic_samples.append({
                            "example_id":example_id,
                            "src_txt": "".join(sentences[topic_left_index:topic_seg_id]),
                            "tgt_txt": labels,
                            "multi_ref": topic_titles
                        })
                    elif strategy == "pool":
                        for other_labels in topic_titles:
                            topic_samples.append({
                                "example_id":example_id,
                                "src_txt": "".join(sentences[topic_left_index:topic_seg_id]),
                                "tgt_txt": other_labels,
                                "multi_ref": topic_titles
                            })
                    else:
                        raise NotImplementedError

                    topic_left_index = topic_seg_id

            for sample_id, topic in enumerate(topic_samples):
                # print(topic)
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

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    print (train_dataset)

    if training_args.do_train:
        num_warmup_steps = 500
        def noam_lambda(current_step: int):
            current_step += 1
            return min(current_step**(-0.5),
                       current_step * num_warmup_steps**(-1.5))

        # 可以在代码修改 configuration 的配置
        max_epochs = int(training_args.num_train_epochs)
        lr = training_args.learning_rate
        batch = training_args.per_device_train_batch_size

        def cfg_modify_fn(cfg):
            cfg.train.lr_scheduler = {
                'type': 'LambdaLR',
                'lr_lambda': noam_lambda,
                'options': {
                    'by_epoch': False
                }
            }
            cfg.preprocessor = {
            "type": "text-gen-tokenizer",
            "sequence_length": 512
            }
            cfg.train.optimizer = {
                "type": "AdamW",
                "lr": lr,
                "options": {}
            }
            cfg.train.hooks[-1] = {
            "type": "EvaluationHook",
            "by_epoch": True,
            "interval": 1
            }
            cfg.train.max_epochs = max_epochs
            cfg.train.dataloader = {
                "batch_size_per_gpu": batch,
                "workers_per_gpu": 1
            }
            return cfg

        kwargs = dict(
            model=model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=WORK_DIR,
            cfg_modify_fn=cfg_modify_fn)
        trainer = build_trainer(
            name=Trainers.text_generation_trainer, default_args=kwargs)
        trainer.train()

    if training_args.do_eval:
        # ---------------------------- Evaluation ---------------------------------
        text_generation = pipeline(Tasks.text_generation,
                                   model=os.path.join(WORK_DIR, "output"),
                                   sequence_length=512)
        m = TextGenerationMetric()
        outputs = {'preds': []}
        inputs = {'tgts': []}

        for sample in eval_dataset:
            output = text_generation(sample['src_txt'].strip())['text']
            outputs['preds'].append(output)
            inputs['tgts'].append(sample['tgt_txt'].strip())

        m.add(outputs, inputs)
        print(f'{m.preds[:10]}\n{m.tgts[:10]}')
        print(m.evaluate())

    if training_args.do_predict:
        # ---------------------------- Inference ---------------------------------
        text_generation = pipeline(Tasks.text_generation, model=os.path.join(WORK_DIR, "output"))

        output_list = []
        for sample in test_dataset:
            output = text_generation(sample['src_txt'])['text']
            sample['tgt_txt'] = output
            output_list.append(json.dumps(sample, ensure_ascii=False) +"\n")
        with open(os.path.join(WORK_DIR, "test_predict_result.txt"), "w") as f:
            f.writelines(output_list)
