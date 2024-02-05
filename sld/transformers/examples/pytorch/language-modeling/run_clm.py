#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import copy
from torch import nn
import evaluate
import jiwer
from itertools import groupby

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--train_dataset_config_names",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_dataset_split_names",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--validation_dataset_config_names",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_dataset_split_names",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--test_dataset_config_names",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--test_dataset_split_names",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=20.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=2.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to train.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to do_predict.",
    )
    parser.add_argument(
        "--vocab_size_speech",
        type=int,
        default=None,
        help="The number of vocab_size_speech.",
    )
    parser.add_argument(
        "--weight_kl_speech",
        type=float,
        default=1.0,
        help="Weight of KL loss for speech",
    )
    parser.add_argument(
        "--weight_ce_speech",
        type=float,
        default=1.0,
        help="Weight of KL loss for speech",
    )
    parser.add_argument(
        "--weight_ce_text",
        type=float,
        default=1.0,
        help="Weight of CE loss for text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="KL loss temperature",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams.",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=150,
        help="max length of text",
    )
    parser.add_argument(
        "--predict_every_epoch",
        action="store_true",
        help="Predict every epoch.",
    )
    parser.add_argument(
        "--time_masking",
        type=float,
        default=0.0,
        help="time masking probability",
    )
    parser.add_argument(
        "--do_init_from_pretrain",
        action="store_true",
        help="Whether to do_init_from_pretrain.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here, and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {"log_with": args.report_to, "project_dir": args.output_dir}

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=None,
            use_auth_token=None,
        )
    else:
        data_files = {}
        extension = "json"
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=None,
            use_auth_token=None,
        )

    if accelerator.is_main_process:
        print(raw_datasets)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    gpt_vocab_size = tokenizer.vocab_size
    text_end = "<text_end>"
    speech_end = "<speech_end>"
    if not args.do_init_from_pretrain and args.do_train:
        tokenizer.add_tokens([text_end, speech_end])
        new_vocab_size = gpt_vocab_size + args.vocab_size_speech + 2
        model.resize_token_embeddings(new_vocab_size)
    if accelerator.is_main_process:
        print('tokenizer.vocab_size', tokenizer.vocab_size)
        print(model)

    speech_end_id = tokenizer.convert_tokens_to_ids(speech_end)
    text_end_id = tokenizer.convert_tokens_to_ids(text_end)
    if accelerator.is_main_process:
        print('speech_end_id', speech_end_id)
        print('text_end_id', text_end_id)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.eos_token_id
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    audio_column_name = "subword_idx" if "subword_idx" in column_names else None

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def tokenize_function(examples):
        output = {'input_ids': [], 'attention_mask': [], 'labels': [], 'raw_labels': []}

        for i in range(len(examples[text_column_name])):
            if examples[audio_column_name][i] and examples[text_column_name][i]:
                speech_code = examples[audio_column_name][i]
                tokens = tokenizer(examples[text_column_name][i].lower())
                max_text_length = args.max_text_length
                if 'opt' in args.model_name_or_path:
                    text_tokens = tokens['input_ids'][1:max_text_length + 1]
                else:
                    text_tokens = tokens['input_ids'][:max_text_length]
                max_speech_length = block_size - 2 - len(text_tokens)
                speech_tokens = [int(x) + gpt_vocab_size + 2 for x in speech_code][:max_speech_length]

                input_idx = speech_tokens + [speech_end_id] + text_tokens + [text_end_id]

                assert len(input_idx) <= block_size
                current_length = len(input_idx)
                padded_input_idx = input_idx + [model.config.pad_token_id] * (block_size - current_length)
                padded_attention_mask = [1] * len(input_idx) + [0] * (block_size - current_length)
                padded_labels = input_idx + [-100] * (block_size - current_length)
                padded_raw_labels = tokens['input_ids'][:block_size] + [model.config.pad_token_id] * (
                        block_size - len(tokens['input_ids']))
                output['input_ids'].append(padded_input_idx)
                output['attention_mask'].append(padded_attention_mask)
                output["labels"].append(padded_labels)
                output['raw_labels'].append(padded_raw_labels)

        return output

    with accelerator.main_process_first():
        tokenized_datasets = DatasetDict()
        tokenized_datasets["train"] = raw_datasets["train"].map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache if accelerator.is_main_process else True,
            new_fingerprint='train_tokenize_text_speech',
            desc="Running tokenizer on dataset",
        )
        tokenized_datasets["validation"] = raw_datasets["validation"].map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache if accelerator.is_main_process else True,
            new_fingerprint='validation_tokenize_text_speech',
            desc="Running tokenizer on dataset",
        )
        tokenized_datasets["test"] = raw_datasets["test"].map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache if accelerator.is_main_process else True,
            new_fingerprint='test_tokenize_text_speech',
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(
            f"Sample {index} of the training set: {tokenizer.convert_ids_to_tokens(train_dataset[index]['input_ids'])}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("clm_no_trainer", experiment_config)
    if accelerator.is_main_process:
        print(experiment_config)

    def predict_dataloader(cur_dataloader, name, epoch, max_step=1000000):
        metric_wer = evaluate.load("./utils/wer")
        metric_cer = evaluate.load("./utils/cer")

        final_all_predictions = []
        final_all_targets = []
        progress_bar_predict = tqdm(range(len(cur_dataloader)), disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(cur_dataloader):
            if step < len(cur_dataloader) - max_step:
                continue
            batch_input_ids = batch['input_ids'].tolist()
            prompt_lens = [x.index(speech_end_id) + 1 for x in batch_input_ids]
            input_ids = []
            for x, prompt_len in zip(batch_input_ids, prompt_lens):
                input_ids.append(x[:prompt_len])
            max_len = max(len(ids) for ids in input_ids)
            padded_input_ids = torch.tensor(
                [[tokenizer.eos_token_id] * (max_len - len(ids)) + ids for ids in input_ids]).to(accelerator.device)
            attention_mask = torch.tensor(
                [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in input_ids]).to(accelerator.device)
            targets = batch['raw_labels']
            with torch.no_grad():
                predictions = model.module.generate(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    max_length=block_size,
                    num_return_sequences=1,
                    num_beams=args.num_beams,
                    use_cache=True,
                    early_stopping=True,
                )
                all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
                accelerator.wait_for_everyone()

                final_predictions = []
                for output in all_predictions.tolist():
                    if text_end_id in output and speech_end_id in output:
                        tokens = output[output.index(speech_end_id) + 1:output.index(text_end_id)]
                    elif speech_end_id in output:
                        tokens = output[output.index(speech_end_id) + 1:]
                    else:
                        tokens = []
                    if tokens:
                        try:
                            text = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
                            text = text.replace('<|endoftext|>', '')
                            text = text.replace('</s>', '')
                        except TypeError:
                            text = ''
                        final_predictions.append(text)
                    else:
                        final_predictions.append('')
                    # print('prediction:', text)
                final_targets = []
                for output in all_targets.tolist():
                    text = tokenizer.decode(output, clean_up_tokenization_spaces=True)
                    text = text.replace('<|endoftext|>', '')
                    text = text.replace('</s>', '')
                    final_targets.append(text)
                    # print('reference:', text)

                metric_wer.add_batch(predictions=final_predictions, references=final_targets)
                metric_cer.add_batch(predictions=final_predictions, references=final_targets)
                final_all_predictions.extend(final_predictions)
                final_all_targets.extend(final_targets)
            progress_bar_predict.update(1)

        accelerator.wait_for_everyone()

        num_samples = len(final_all_targets)
        wer_score = metric_wer.compute() * 100
        cer_score = metric_cer.compute() * 100

        if accelerator.is_main_process:
            output_prediction_file = os.path.join(args.output_dir, f"epoch_{epoch}",
                                                  f"{name}_generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(final_all_predictions))
            output_labels_file = os.path.join(args.output_dir, f"epoch_{epoch}", f"{name}_decoded_labels.txt")
            with open(output_labels_file, "w") as writer:
                writer.write("\n".join(final_all_targets))
            logger.info("*** Predict ***")
            logger.info(f"set: {name} epoch {epoch} wer: {wer_score} cer: {cer_score} num_samples: {num_samples}")
            with open(os.path.join(args.output_dir, f"epoch_{epoch}", f"all_results_{name}.json"), "w") as f:
                json.dump({"set": name, "epoch": epoch, "wer": wer_score, "cer": cer_score, "num_samples": num_samples},
                          f)

        return wer_score, cer_score

    best_dev_wer_score = 100000
    best_epoch = 0

    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        starting_epoch = 0
        best_epoch = 0

        resume_step = None
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch
        print('args.weight_ce_speech', args.weight_ce_speech)
        print('args.weight_ce_text', args.weight_ce_text)
        print('args.weight_kl_speech', args.weight_kl_speech)

        def all_loss(batch, model):
            if args.time_masking > 0.0:
                mask = torch.rand(batch['input_ids'].shape) < args.time_masking
                batch['input_ids'][mask] = tokenizer.eos_token_id

            labels = batch['labels']
            batch.pop('raw_labels')
            outputs = model(**batch)
            logits = outputs.logits
            ce_loss_fct_speech = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss_fct_text = nn.CrossEntropyLoss(ignore_index=-100)
            kl_loss_fct_speech = nn.KLDivLoss(reduction="batchmean")
            # Shift so that tokens < n predict n
            mask = batch['attention_mask']
            eps = 1e-9
            shift_logits = logits[:, :-1, -args.vocab_size_speech:].contiguous() * mask[:, :-1].unsqueeze(
                -1) + eps
            num_classes = args.vocab_size_speech
            temp = (labels[:, 1:] - gpt_vocab_size - 2) * mask[:, 1:]
            temp[temp < 0] = 0
            one_hot = nn.functional.one_hot(temp, num_classes=num_classes)
            epsilon = 0.1
            shift_t_logits = one_hot * (1 - epsilon) + epsilon / num_classes
            shift_t_logits = shift_t_logits * mask[:, 1:].unsqueeze(-1) + eps
            loss_kl_speech = (
                    kl_loss_fct_speech(
                        nn.functional.log_softmax(shift_logits / temperature, dim=-1),
                        nn.functional.softmax(shift_t_logits / temperature, dim=-1),
                    )
                    * temperature ** 2
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels_text = torch.where(shift_labels >= gpt_vocab_size + 1, -100, shift_labels)
            loss_ce_text = ce_loss_fct_text(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_text.view(-1))
            shift_labels_speech = torch.where(shift_labels < gpt_vocab_size + 1, -100, shift_labels)
            loss_ce_speech = ce_loss_fct_speech(shift_logits.view(-1, shift_logits.size(-1)),
                                                shift_labels_speech.view(-1))

            loss = args.weight_ce_speech * loss_ce_speech \
                   + args.weight_ce_text * loss_ce_text \
                   + args.weight_kl_speech * loss_kl_speech

            return loss, loss_ce_speech, loss_ce_text, loss_kl_speech

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            total_loss_ce_speech = 0
            total_loss_ce_text = 0
            total_loss_kl_speech = 0
            eval_loss = 0
            eval_loss_ce_speech = 0
            eval_loss_ce_text = 0
            eval_loss_kl_speech = 0

            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                temperature = args.temperature
                with accelerator.accumulate(model):
                    loss, loss_ce_speech, loss_ce_text, loss_kl_speech = all_loss(batch, model)
                    # We keep track of the loss at each epoch
                    total_loss += loss.detach().float()
                    total_loss_ce_speech += loss_ce_speech.detach().float()
                    total_loss_ce_text += loss_ce_text.detach().float()
                    total_loss_kl_speech += loss_kl_speech.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            losses = []
            losses_ce_speech = []
            losses_ce_text = []
            losses_kl_speech = []
            max_step = 10 * args.gradient_accumulation_steps
            for step, batch in enumerate(eval_dataloader):
                if step < len(eval_dataloader) - max_step:
                    continue
                with torch.no_grad():
                    loss, loss_ce_speech, loss_ce_text, loss_kl_speech = all_loss(batch, model)
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                    losses_ce_speech.append(
                        accelerator.gather_for_metrics(loss_ce_speech.repeat(args.per_device_eval_batch_size)))
                    losses_ce_text.append(
                        accelerator.gather_for_metrics(loss_ce_text.repeat(args.per_device_eval_batch_size)))
                    losses_kl_speech.append(
                        accelerator.gather_for_metrics(loss_kl_speech.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            losses_ce_speech = torch.cat(losses_ce_speech)
            losses_ce_text = torch.cat(losses_ce_text)
            losses_kl_speech = torch.cat(losses_kl_speech)
            try:
                eval_loss = torch.mean(losses)
                eval_loss_ce_speech = torch.mean(losses_ce_speech)
                eval_loss_ce_text = torch.mean(losses_ce_text)
                eval_loss_kl_speech = torch.mean(losses_kl_speech)
                perplexity = math.exp(eval_loss_ce_text)
            except OverflowError:
                perplexity = float("inf")

            logger.info(
                f"epoch {epoch}: text perplexity: {perplexity} eval_loss: {eval_loss} eval_loss_ce_speech: {eval_loss_ce_speech} eval_loss_ce_text: {eval_loss_ce_text} eval_loss_kl_speech: {eval_loss_kl_speech}")

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

            dev_wer_score = 100
            dev_cer_score = 100
            test_wer_score = 100
            test_cer_score = 100
            if args.predict_every_epoch:
                dev_wer_score, dev_cer_score = predict_dataloader(eval_dataloader, 'dev', epoch,
                                                                  max_step=10 * args.gradient_accumulation_steps)
                test_wer_score, test_cer_score = predict_dataloader(test_dataloader, 'test', epoch,
                                                                    max_step=10 * args.gradient_accumulation_steps)
                if dev_wer_score < best_dev_wer_score:
                    best_epoch = epoch
                    best_dev_wer_score = dev_wer_score

            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss.item(),
                    "eval_loss_ce_text": eval_loss_ce_text.item(),
                    "eval_loss_ce_speech": eval_loss_ce_speech.item(),
                    "eval_loss_kl_speech": eval_loss_kl_speech.item(),
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "train_loss_ce_speech": total_loss_ce_speech.item() / len(train_dataloader),
                    "train_loss_ce_text": total_loss_ce_text.item() / len(train_dataloader),
                    "train_loss_kl_speech": total_loss_kl_speech.item() / len(train_dataloader),
                    "dev_wer_score": dev_wer_score,
                    "dev_cer_score": dev_cer_score,
                    "test_wer_score": test_wer_score,
                    "test_cer_score": test_cer_score,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

            accelerator.end_training()

    if args.do_predict:
        accelerator.print(f"Resumed from checkpoint of best epoch: {best_epoch} best_dev_wer: {best_dev_wer_score}")
        accelerator.load_state(os.path.join(args.output_dir, f"epoch_{best_epoch}"))
        output_dir = f"epoch_{best_epoch}_best"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

        logger.info("*** Predict ***")
        predict_dataloader(eval_dataloader, 'dev', f"{best_epoch}_best")
        predict_dataloader(test_dataloader, 'test', f"{best_epoch}_best")

        for name in ['dev', 'test']:
            output_prediction_file = os.path.join(args.output_dir, f"epoch_{best_epoch}_best",
                                                  f"{name}_generated_predictions.txt")
            output_labels_file = os.path.join(args.output_dir, f"epoch_{best_epoch}_best", f"{name}_decoded_labels.txt")
            if name == 'dev':
                split_num = 2703
            else:
                split_num = 2620
            refs_clean = []
            hyps_clean = []
            refs_other = []
            hyps_other = []
            with open(output_labels_file, 'r') as f_r:
                with open(output_prediction_file, 'r') as f_h:
                    for idx, (line_r, line_h) in enumerate(zip(f_r, f_h)):
                        if idx < int(split_num):
                            refs_clean.append(line_r.strip())
                            hyps_clean.append(line_h.strip())
                        else:
                            refs_other.append(line_r.strip())
                            hyps_other.append(line_h.strip())

            # compute WER
            if refs_clean:
                wer_clean = jiwer.wer(refs_clean, hyps_clean) * 100
                cer_clean = jiwer.cer(refs_clean, hyps_clean) * 100
            else:
                wer_clean = 100
                cer_clean = 100
            if refs_other:
                wer_other = jiwer.wer(refs_other, hyps_other) * 100
                cer_other = jiwer.cer(refs_other, hyps_other) * 100
            else:
                wer_other = 100
                cer_other = 100
            num_samples_clean = len(refs_clean)
            num_samples_other = len(refs_other)

            logger.info(
                f"set: {name} epoch {best_epoch} num_samples_clean: {num_samples_clean} num_samples_other: {num_samples_other} wer_clean: {wer_clean} wer_other: {wer_other} cer_clean: {cer_clean} cer_other: {cer_other} ")
            with open(os.path.join(args.output_dir, f"epoch_{best_epoch}_best", f"all_results_{name}.json"), "w") as f:
                json.dump({"set": name, "epoch": best_epoch, "num_samples_clean": num_samples_clean,
                           "num_samples_other": num_samples_other, "wer_clean": wer_clean,
                           "wer_other": wer_other, "cer_clean": cer_clean, "cer_other": cer_other}, f)


if __name__ == "__main__":
    main()
