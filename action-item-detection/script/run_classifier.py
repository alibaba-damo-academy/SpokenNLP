# coding=utf-8
# Copyright (c) 2023, Alibaba Group;
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import re
import modeling
import optimization
import tokenization
import tensorflow as tf

import pickle
import logging
import random
import numpy as np
from sklearn.metrics import classification_report

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "record_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "predict_dir", None,
    "The prediction directory where the prediction files will be written.")

flags.DEFINE_string(
    "best_model_dir", None,
    "The best model directory where the best model checkpoints will be written")

flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_label_smoothing", False,
    "Whether to do the label smoothing to compute the loss."
)

flags.DEFINE_string(
    "loss_type", "loss",
    "The type of the loss, including loss, focal loss."
)

flags.DEFINE_string(
    "classifier_input", "cls",
    "The type of the classifier input, including cls, sep, token_avg, token_max."
)

flags.DEFINE_string(
    "classifier_model", "linear",
    "The type of the classifier model, including linear."
)

flags.DEFINE_string(
    "context_type", "sentence",
    "The type of the context, including sentence, left, right, context."
)

flags.DEFINE_string(
    "dev_context_type", "sentence",
    "The type of the context (dev), including sentence, left, right, context."
)

flags.DEFINE_string(
    "test_context_type", "sentence",
    "The type of the context (test), including sentence, left, right, context."
)

flags.DEFINE_string(
    "noisy_type", "remain",
    "Type to skip/remain/update noisy label data (i.e., negative focus sentence with positive context)."
)

flags.DEFINE_integer(
    "context_width", 1,
    "The width of the context."
)

flags.DEFINE_float(
    "threshold", 0.5,
    "The threshold to classify positive example and negative example."
)

flags.DEFINE_string("train_file", 'train.txt',
                    "The train file that the BERT model was trained on.")

flags.DEFINE_string("dev_file", 'dev.txt',
                    "The evaluation file that the BERT model was evaluated on.")

flags.DEFINE_string("test_file", 'test.txt',
                    "The test file that the BERT model was tested on.")

flags.DEFINE_string("test_type", "file",
                    "The test type, including file and dir.")

flags.DEFINE_string("test_path", None,
                    "The test fiel or directory where the BERT model will be tested on.")

flags.DEFINE_string("model_log_file", "model.log.txt",
                    "The log file of model training and processing.")

flags.DEFINE_bool("do_export", False, "Whether to export the model.")

flags.DEFINE_bool("do_frozen", False, "Whether to frozen the graph.")

flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate.")

flags.DEFINE_float("kl_alpha", 4.0, "KL loss alpha weight.")

flags.DEFINE_string("drop_type", "r-drop", "Drop type, R-Drop, Context-Drop-Dynamic.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_integer("back_up_num", 5, "The number of best model back up.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train", 100.0,
                   "Total number of training steps or epochs to perform.")

flags.DEFINE_string("num_train_type", "epoch",
                    "The type of training number, step or epoch")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float("best_f1", -0.1, "Best f1 score value.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("train_summary_steps", 100,
                     "How many steps to print and summary metrics for training")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_sentence, context_left=None, context_right=None, context_global=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_sentence: string. The focus sentence for action item detection.
          context_left: (Optional) string list. The left context of the focus sentence.
          context_right: (Optional) string list. The right context of the focus sentence.
          context_global: (Optional) string list. The global context of the focus sentence.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_sentence = text_sentence
        self.context_left = context_left
        self.context_right = context_right
        self.context_global = context_global
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MeetProcessor(DataProcessor):
    """
    Processor for the Meeting data set.
    """

    def _read_txt(self, data_file):
        line_dict_list = []
        context_sep = "###"
        fields = ["sentence", "label", "line_id", "sentence_id", "document_length",
                  "left_context", "right_context", "global_context"]
        with tf.gfile.Open(data_file, "r") as fr:
            for line in fr.readlines():
                line_dict = {}
                line_list = line.strip().split("\t")
                line_dict["sentence"] = line_list[fields.index("sentence")]
                line_dict["label"] = line_list[fields.index("label")]
                line_dict["line_id"] = line_list[fields.index("line_id")]
                line_dict["sentence_id"] = line_list[fields.index("sentence_id")]
                line_dict["document_length"] = line_list[fields.index("document_length")]
                line_dict["left_context"] = line_list[fields.index("left_context")].strip().strip(context_sep).split(context_sep) if "left_context" in fields else []
                line_dict["right_context"] = line_list[fields.index("right_context")].strip().strip(context_sep).split(context_sep) if "right_context" in fields else []
                line_dict["global_context"] = line_list[fields.index("global_context")].strip().strip(context_sep).split(context_sep) if "global_context" in fields else []
                line_dict_list.append(line_dict)
        return line_dict_list

    def _get_context(self, context_list, context_width=None, context_label_sep="@@@"):
        default_context_width = FLAGS.context_width
        context_width = default_context_width if context_width is None else context_width

        context_label_list = []
        for context in context_list:
            context_label = context.strip().split(context_label_sep)
            context_item = {"sentence": "", "label": "0"}
            if len(context_label) == 2:
                sentence = context_label[0].strip()
                label = context_label[1]
                context_item = {"sentence": sentence, "label": label}
            context_label_list.append(context_item)

        context = []
        if len(context_list) >= context_width:
            context.extend(context_label_list[:context_width])
        else:
            padding_item = {"sentence": "", "label": "0"}
            padding_list = [padding_item for _ in range(context_width - len(context_list))]
            context.extend(context_label_list)
            context.extend(padding_list)

        for context_item in context:
            context_item["sentence"] = tokenization.convert_to_unicode(context_item["sentence"])
            context_item["label"] = tokenization.convert_to_unicode(context_item["label"])

        return context

    def _keep_context(self, rng, threshold=0.5):
        return rng.random() >= threshold

    def _create_examples(self, data_file, set_type):
        lines = self._read_txt(data_file)
        examples = []
        rng = random.Random(2022)
        example_id = 0
        for (i, line_dict) in enumerate(lines):
            text_sentence = tokenization.convert_to_unicode(line_dict["sentence"])
            context_left_span = line_dict["left_context"]
            context_right_span = line_dict["right_context"]
            context_global_span = line_dict["global_context"]
            label = tokenization.convert_to_unicode(line_dict["label"])

            if set_type == "train":
                input_types = FLAGS.context_type.strip().split("+")

                if FLAGS.drop_type.lower() == "none":
                    context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                    context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                    context_global_list = self._get_context(context_global_span) if "global" in input_types else []
                    context_left = [context_item["sentence"] for context_item in context_left_list]
                    context_right = [context_item["sentence"] for context_item in context_right_list]
                    context_global = [context_item["sentence"] for context_item in context_global_list]
                    sequence_label = label
                    if any([context_item["label"] == "1" for context_item in context_left_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_right_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_global_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if sequence_label != label:
                        if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                            continue
                        elif FLAGS.noisy_type.lower() == "update":  # update noisy label
                            label = sequence_label
                    guid = "{}-{}".format(set_type, example_id)
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                                 context_right=context_right, context_global=context_global,
                                                 label=label))
                    example_id += 1

                # R-Drop
                elif FLAGS.drop_type.lower() == "r-drop":
                    context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                    context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                    context_global_list = self._get_context(context_global_span) if "global" in input_types else []
                    context_left = [context_item["sentence"] for context_item in context_left_list]
                    context_right = [context_item["sentence"] for context_item in context_right_list]
                    context_global = [context_item["sentence"] for context_item in context_global_list]
                    sequence_label = label
                    if any([context_item["label"] == "1" for context_item in context_left_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_right_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_global_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if sequence_label != label:
                        if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                            continue
                        elif FLAGS.noisy_type.lower() == "update":  # update noisy label
                            label = sequence_label
                    guid = "{}-{}".format(set_type, example_id)
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                                 context_right=context_right, context_global=context_global,
                                                 label=label))
                    example_id += 1
                    guid = "{}-{}".format(set_type, example_id)
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                                 context_right=context_right, context_global=context_global,
                                                 label=label))
                    example_id += 1

                # Context-Drop-Fix
                elif FLAGS.drop_type.lower() == "context-drop-fix":
                    guid = "{}-{}".format(set_type, example_id)
                    context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                    context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                    context_global_list = self._get_context(context_global_span) if "global" in input_types else []
                    context_left = [context_item["sentence"] for context_item in context_left_list]
                    context_right = [context_item["sentence"] for context_item in context_right_list]
                    context_global = [context_item["sentence"] for context_item in context_global_list]
                    sequence_label = label
                    if any([context_item["label"] == "1" for context_item in context_left_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_right_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in context_global_list]):
                        sequence_label = tokenization.convert_to_unicode("1")
                    if sequence_label != label:
                        if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                            continue
                        elif FLAGS.noisy_type.lower() == "update":  # update noisy label
                            label = sequence_label
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                                 context_right=context_right, context_global=context_global,
                                                 label=label))
                    example_id += 1
                    guid = "{}-{}".format(set_type, example_id)
                    context_left, context_right, context_global = [], [], []
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                                 context_right=context_right, context_global=context_global,
                                                 label=label))
                    example_id += 1

                # Context-Drop-Dynamic
                elif FLAGS.drop_type.lower() == "context-drop-dynamic":
                    context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                    context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                    context_global_list = self._get_context(context_global_span) if "global" in input_types else []

                    first_context_left_list = [context_item for context_item in context_left_list if self._keep_context(rng, threshold=0.5)]
                    first_context_right_list = [context_item for context_item in context_right_list if self._keep_context(rng, threshold=0.5)]
                    first_context_global_list = [context_item for context_item in context_global_list if self._keep_context(rng, threshold=0.5)]
                    first_context_left = [context_item["sentence"] for context_item in first_context_left_list]
                    first_context_right = [context_item["sentence"] for context_item in first_context_right_list]
                    first_context_global = [context_item["sentence"] for context_item in first_context_global_list]
                    first_label = label
                    if any([context_item["label"] == "1" for context_item in first_context_left_list]):
                        first_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in first_context_right_list]):
                        first_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in first_context_global_list]):
                        first_label = tokenization.convert_to_unicode("1")

                    second_context_left_list = [context_item for context_item in context_left_list if self._keep_context(rng, threshold=0.5)]
                    second_context_right_list = [context_item for context_item in context_right_list if self._keep_context(rng, threshold=0.5)]
                    second_context_global_list = [context_item for context_item in context_global_list if self._keep_context(rng, threshold=0.5)]
                    second_context_left = [context_item["sentence"] for context_item in second_context_left_list]
                    second_context_right = [context_item["sentence"] for context_item in second_context_right_list]
                    second_context_global = [context_item["sentence"] for context_item in second_context_global_list]
                    second_label = label
                    if any([context_item["label"] == "1" for context_item in second_context_left_list]):
                        second_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in second_context_right_list]):
                        second_label = tokenization.convert_to_unicode("1")
                    if any([context_item["label"] == "1" for context_item in second_context_global_list]):
                        second_label = tokenization.convert_to_unicode("1")

                    if first_label != label or second_label != label:
                        if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                            continue
                        elif FLAGS.noisy_type.lower() == "remain":  # remain noisy label
                            first_label = label
                            second_label = label

                    guid = "{}-{}".format(set_type, example_id)
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=first_context_left,
                                                 context_right=first_context_right, context_global=first_context_global,
                                                 label=first_label))
                    example_id += 1
                    guid = "{}-{}".format(set_type, example_id)
                    examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=second_context_left,
                                                 context_right=second_context_right, context_global=second_context_global,
                                                 label=second_label))
                    example_id += 1

            elif set_type == "dev":
                guid = "{}-{}".format(set_type, i)
                input_types = FLAGS.dev_context_type.strip().split("+")
                context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                context_global_list = self._get_context(context_global_span) if "global" in input_types else []
                context_left = [context_item["sentence"] for context_item in context_left_list]
                context_right = [context_item["sentence"] for context_item in context_right_list]
                context_global = [context_item["sentence"] for context_item in context_global_list]
                # sequence_label = label
                # if any([context_item["label"] == "1" for context_item in context_left_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if any([context_item["label"] == "1" for context_item in context_right_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if any([context_item["label"] == "1" for context_item in context_global_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if sequence_label != label:
                #     if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                #         continue
                #     elif FLAGS.noisy_type.lower() == "update":  # update noisy label
                #         label = sequence_label
                examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                             context_right=context_right, context_global=context_global,
                                             label=label))

            elif set_type == "test":
                guid = "{}-{}".format(set_type, i)
                input_types = FLAGS.test_context_type.strip().split("+")
                context_left_list = self._get_context(context_left_span) if "left" in input_types else []
                context_right_list = self._get_context(context_right_span) if "right" in input_types else []
                context_global_list = self._get_context(context_global_span) if "global" in input_types else []
                context_left = [context_item["sentence"] for context_item in context_left_list]
                context_right = [context_item["sentence"] for context_item in context_right_list]
                context_global = [context_item["sentence"] for context_item in context_global_list]
                # sequence_label = label
                # if any([context_item["label"] == "1" for context_item in context_left_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if any([context_item["label"] == "1" for context_item in context_right_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if any([context_item["label"] == "1" for context_item in context_global_list]):
                #     sequence_label = tokenization.convert_to_unicode("1")
                # if sequence_label != label:
                #     if FLAGS.noisy_type.lower() == "skip":  # skip noisy
                #         continue
                #     elif FLAGS.noisy_type.lower() == "update":  # update noisy label
                #         label = sequence_label
                examples.append(InputExample(guid=guid, text_sentence=text_sentence, context_left=context_left,
                                             context_right=context_right, context_global=context_global,
                                             label=label))

        return examples

    def get_train_examples(self, data_dir):
        return self._create_examples(FLAGS.train_file, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(FLAGS.dev_file, "dev")

    def get_test_examples(self, test_file):
        return self._create_examples(test_file, "test")

    def get_labels(self):
        return ["0", "1"]


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, dataset_type="train"):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # Add 0917
    output_label2id_file = os.path.join(FLAGS.output_dir, "label2id.pkl")
    if not os.path.exists(output_label2id_file):
        with open(output_label2id_file, "wb") as fw:
            pickle.dump(label_map, fw)
    # Add 0917 END

    tokens_sentence = tokenizer.tokenize(example.text_sentence)
    tokens_left_list = []  # [[]] keep [SEP] when no context, [] remove [SEP] when no context
    if FLAGS.classifier_input == "sep":
        tokens_left_list = [[]]
    if example.context_left:
        tokens_left_list = [tokenizer.tokenize(text_left) for text_left in example.context_left]
    tokens_right_list = []  # [[]] keep [SEP] when no context, [] remove [SEP] when no context
    if FLAGS.classifier_input == "sep":
        tokens_right_list = [[]]
    if example.context_right:
        tokens_right_list = [tokenizer.tokenize(text_right) for text_right in example.context_right]
    tokens_global_list = []  # [[]] keep [SEP] when no context, [] remove [SEP] when no context
    if FLAGS.classifier_input == "sep":
        tokens_global_list = [[]]
    if example.context_global:
        tokens_global_list = [tokenizer.tokenize(text_global) for text_global in example.context_global]

    context_type = FLAGS.context_type
    context_width = FLAGS.context_width

    input_types = context_type.strip().split("+")
    if dataset_type == "dev":
        input_types = FLAGS.dev_context_type.strip().split("+")
    elif dataset_type == "test":
        input_types = FLAGS.test_context_type.strip().split("+")

    if len(input_types) == 1 and "sentence" in input_types:
        if len(tokens_sentence) > max_seq_length - 2:
            tokens_sentence = tokens_sentence[0:(max_seq_length - 2)]
    else:
        max_length = max_seq_length - (len(input_types) - 1) * context_width - 2
        _truncate_seq_pair(tokens_sentence, tokens_left_list, tokens_right_list, tokens_global_list, max_length)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_id = 0
    segment_ids.append(segment_id)

    for input_type in input_types:
        if input_type == "sentence":
            for token in tokens_sentence:
                tokens.append(token)
                segment_ids.append(segment_id)
            tokens.append("[SEP]")
            segment_ids.append(segment_id)
            segment_id = 1 - segment_id

        elif input_type == "left":
            for tokens_left in reversed(tokens_left_list):
                if tokens_left == [] and FLAGS.classifier_input != "sep":
                    continue
                for token in tokens_left:
                    tokens.append(token)
                    segment_ids.append(segment_id)
                tokens.append("[SEP]")
                segment_ids.append(segment_id)
                segment_id = 1 - segment_id

        elif input_type == "right":
            for tokens_right in tokens_right_list:
                if tokens_right == [] and FLAGS.classifier_input != "sep":
                    continue
                for token in tokens_right:
                    tokens.append(token)
                    segment_ids.append(segment_id)
                tokens.append("[SEP]")
                segment_ids.append(segment_id)
                segment_id = 1 - segment_id

        elif input_type == "global":
            for tokens_global in tokens_global_list:
                if tokens_global == [] and FLAGS.classifier_input != "sep":
                    continue
                for token in tokens_global:
                    tokens.append(token)
                    segment_ids.append(segment_id)
                tokens.append("[SEP]")
                segment_ids.append(segment_id)
                segment_id = 1 - segment_id

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, dataset_type="train"):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, dataset_type)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            # d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_sentence, tokens_left_list, tokens_right_list, tokens_global_list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        left_length = sum([len(tokens_left) for tokens_left in tokens_left_list])
        right_length = sum([len(tokens_right) for tokens_right in tokens_right_list])
        global_length = sum([len(tokens_global) for tokens_global in tokens_global_list])
        total_length = len(tokens_sentence) + left_length + right_length + global_length

        if total_length <= max_length:
            break

        if left_length > 0 or right_length > 0 or global_length > 0:
            if left_length >= right_length and left_length >= global_length:
                current_i = len(tokens_left_list) - 1
                for i in range(len(tokens_left_list) - 1, -1, -1):
                    if len(tokens_left_list[i]) > 0:
                        current_i = i
                        break
                tokens_left_list[current_i].pop(0)
            elif right_length >= left_length and right_length >= global_length:
                current_i = len(tokens_right_list) - 1
                for i in range(len(tokens_right_list) - 1, -1, -1):
                    if len(tokens_right_list[i]) > 0:
                        current_i = i
                        break
                tokens_right_list[current_i].pop()
            else:
                current_i = len(tokens_global_list) - 1
                for i in range(len(tokens_global_list) - 1, -1, -1):
                    if len(tokens_global_list[i]) > 0:
                        current_i = i
                        break
                tokens_global_list[current_i].pop()
        else:
            tokens_sentence.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    hidden_size = model.get_pooled_output().shape[-1].value

    if FLAGS.classifier_input == "cls":
        # output_layer: [batch_size, hidden_size]
        output_layer = model.get_pooled_output()

    elif FLAGS.classifier_input == "sep":
        # [batch_size, sequence_length, hidden_size]
        sequence_outputs = model.get_sequence_output()

        sep_token_id = 102  # [SEP] 102
        focus_ind = tf.where(tf.equal(input_ids, sep_token_id))
        context_type = FLAGS.context_type
        context_width = FLAGS.context_width

        input_types = context_type.strip().split("+")
        if not is_training:
            input_types = FLAGS.test_context_type.strip().split("+")

        sentence_sep_id = input_types.index("sentence") * context_width

        if len(input_types) == 3:
            focus_ind = tf.reshape(focus_ind, [-1, 2 * context_width + 1, 2])[:, sentence_sep_id, :]
        if len(input_types) == 2:
            focus_ind = tf.reshape(focus_ind, [-1, context_width + 1, 2])[:, sentence_sep_id, :]
        if len(input_types) == 1:
            focus_ind = tf.reshape(focus_ind, [-1, 1, 2])[:, sentence_sep_id, :]
        else:
            focus_ind = tf.reshape(focus_ind, [-1, 1, 2])[:, 0, :]

        output_layer = tf.gather_nd(sequence_outputs, focus_ind)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])

    elif FLAGS.classifier_input == "token_avg":
        # [batch_size, sequence_length - 1, hidden_size]
        sequence_outputs = model.get_sequence_output()[:, 1:, :]

        avg_pooling_mask = input_mask[:, 1:]   # [batch_size, sequence_length - 1]
        avg_pooling_mask = tf.expand_dims(avg_pooling_mask, axis=-1)  # [batch_size, sequence_length - 1, 1]
        avg_pooling_mask = tf.cast(avg_pooling_mask, tf.float32)

        focus_mask = segment_ids[:, 1:]   # [batch_size, sequence_length - 1]
        focus_mask = tf.expand_dims(focus_mask, axis=-1)   # [batch_size, sequence_length - 1, 1]
        focus_mask = tf.cast(focus_mask, tf.float32)
        focus_mask = 1.0 - focus_mask
        avg_pooling_mask = avg_pooling_mask * focus_mask

        avg_pooling_length = tf.maximum(tf.reduce_sum(avg_pooling_mask, axis=1), 1.0)
        sequence_outputs = sequence_outputs * avg_pooling_mask
        output_layer = tf.reduce_sum(sequence_outputs, axis=1) / avg_pooling_length
        output_layer = tf.reshape(output_layer, [-1, hidden_size])

    elif FLAGS.classifier_input == "token_max":
        # [batch_size, sequence_length - 1, hidden_size]
        sequence_outputs = model.get_sequence_output()[:, 1:, :]

        max_pooling_mask = input_mask[:, 1:]   # [batch_size, sequence_length - 1]
        max_pooling_mask = tf.expand_dims(max_pooling_mask, axis=-1)  # [batch_size, sequence_length - 1, 1]
        max_pooling_mask = tf.cast(max_pooling_mask, tf.float32)

        focus_mask = segment_ids[:, 1:]  # [batch_size, sequence_length - 1]
        focus_mask = tf.expand_dims(focus_mask, axis=-1)  # [batch_size, sequence_length - 1, 1]
        focus_mask = tf.cast(focus_mask, tf.float32)
        focus_mask = 1.0 - focus_mask
        max_pooling_mask = max_pooling_mask * focus_mask

        max_pooling_mask = (1.0 - max_pooling_mask) * -10000.0
        sequence_outputs = sequence_outputs + max_pooling_mask
        output_layer = tf.reduce_max(sequence_outputs, axis=1)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])

    else:
        # output_layer: [batch_size, hidden_size]
        output_layer = model.get_pooled_output()

    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=1 - FLAGS.dropout_rate)

    if FLAGS.classifier_model == "linear":  # todo: architecture
        # output_weights: [num_labels, hidden_size]
        classifier_input_size = hidden_size
        output_weights = tf.get_variable(
            "output_weights", [num_labels, classifier_input_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        # output_bias: [num_labels]
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
        # logits: [batch_size, num_labels]
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
    else:
        # output_weights: [num_labels, hidden_size]
        classifier_input_size = hidden_size
        output_weights = tf.get_variable(
            "output_weights", [num_labels, classifier_input_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        # output_bias: [num_labels]
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
        # logits: [batch_size, num_labels]
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    def kl(x, y):
        x = tf.nn.softmax(x, axis=-1)
        y = tf.nn.softmax(y, axis=-1)
        X = tf.distributions.Categorical(probs=x)
        Y = tf.distributions.Categorical(probs=y)
        return tf.distributions.kl_divergence(X, Y)

    with tf.variable_scope("loss"):
        # probabilities: [batch_size, num_labels]
        probabilities = tf.nn.softmax(logits, axis=-1, name="pred_prob")

        pred_probs = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # label smoothing
        if FLAGS.do_label_smoothing:
            epsilon = 0.1
            one_hot_labels = (1 - epsilon) * one_hot_labels + (epsilon / num_labels)

        # loss computing
        if FLAGS.loss_type == "focal_loss":  # focal loss
            gamma = 2
            per_example_loss = -tf.reduce_sum(one_hot_labels * ((1 - pred_probs) ** gamma) * log_probs, axis=-1)
        else:   # loss
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        # loss = tf.reduce_mean(per_example_loss)
        loss_ce = tf.reduce_mean(per_example_loss)
        if not is_training:
            loss = loss_ce
        elif FLAGS.drop_type.lower() == "none":
            loss = loss_ce
        else:
            alpha = FLAGS.kl_alpha
            contrastive_logits = tf.reshape(logits, [-1, 2, num_labels])
            logits1 = contrastive_logits[:, 0, :]  # [batch_size / 2, num_labels]
            logits2 = contrastive_logits[:, 1, :]  # [batch_size / 2, num_labels]
            per_example_loss_kl = kl(logits1, logits2) + kl(logits2, logits1)
            loss_kl = tf.reduce_mean(per_example_loss_kl) / 2
            loss = loss_ce + alpha * loss_kl

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, num_train_examples, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        logging.info("labels_ids shape: {}".format(label_ids.get_shape()))
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("total loss shape: {}".format(total_loss.get_shape()))
            train_op, new_global_step = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            new_global_epoch = tf.cast(new_global_step, tf.float32) * FLAGS.train_batch_size / (num_train_examples + 1e-8)

            # predictions = tf.argmax(logits, axis=-1, name="predictions")
            predictions = tf.cast(probabilities[:, 1] > FLAGS.threshold, dtype=tf.int32)

            pred_labels = tf.cast(predictions, tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(label_ids, pred_labels), tf.float32))

            tp = tf.reduce_sum(tf.cast(label_ids * pred_labels, tf.float32))
            fp = tf.reduce_sum(tf.cast((1 - label_ids) * pred_labels, tf.float32))
            fn = tf.reduce_sum(tf.cast(label_ids * (1 - pred_labels), tf.float32))
            tn = tf.reduce_sum(tf.cast((1 - label_ids) * (1 - pred_labels), tf.float32))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1score = (2 * precision * recall) / (precision + recall + 1e-8)
            confusion = tf.convert_to_tensor([tp, fp, fn, tn])

            logging_hook = tf.train.LoggingTensorHook({"10_loss": total_loss,
                                                       "01_global_step": new_global_step,
                                                       "02_global_epoch": new_global_epoch,
                                                       "03_label": label_ids,
                                                       "04_predict": predictions,
                                                       "05_f1": f1score,
                                                       "06_precision": precision,
                                                       "07_recall": recall,
                                                       "08_accuracy": accuracy,
                                                       "09_confusion": confusion,
                                                       },
                                                      every_n_iter=FLAGS.train_summary_steps)

            tf.summary.scalar("train/accuracy", accuracy)
            tf.summary.scalar("train/f1", f1score)
            tf.summary.scalar("train/precision", precision)
            tf.summary.scalar("train/recall", recall)
            tf.summary.scalar("train/loss", total_loss)

            summary_hook = tf.train.SummarySaverHook(
                save_steps=FLAGS.train_summary_steps,
                output_dir=FLAGS.output_dir,
                scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook, summary_hook],
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                predictions = tf.cast(probabilities[:, 1] > FLAGS.threshold, dtype=tf.int32)

                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
                auc = tf.metrics.auc(labels=label_ids, predictions=predictions, weights=is_real_example)
                precision = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)
                recall = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)
                f1score = (2 * precision[0] * recall[0] / (precision[0] + recall[0] + 1e-8), precision[1])

                return {
                    "eval/accuracy": accuracy,
                    "eval/loss": loss,
                    "eval/precision": precision,
                    "eval/recall": recall,
                    "eval/f1": f1score,
                }

            eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
            # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
            )

        else:
            # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predictions = tf.cast(probabilities[:, 1] > FLAGS.threshold, dtype=tf.int32)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "probabilities": probabilities,
                    "predictions": predictions,
                }
            )

        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            # d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
    input_ids = tf.placeholder(tf.int32, name="input_ids")
    input_mask = tf.placeholder(tf.int32, name="input_mask")
    segment_ids = tf.placeholder(tf.int32, name="segment_ids")
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "label_ids": label_ids,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
    })()
    return input_fn


def check_path(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


def main(_):
    # tf.logging.set_verbosity(tf.logging.INFO)
    check_path(FLAGS.output_dir)
    check_path(FLAGS.best_model_dir)
    check_path(FLAGS.record_dir)

    logging_file = os.path.join(FLAGS.output_dir, FLAGS.model_log_file)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(logging_file, mode="a"),
                                  logging.StreamHandler()
                                  ])

    processors = {
        "meet": MeetProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # tf.gfile.MakeDirs(FLAGS.output_dir)
    # best_f1 = FLAGS.best_f1
    # tf.gfile.MakeDirs(FLAGS.best_model_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    # Add 0917
    id_to_label = {}
    for i, label in enumerate(label_list):
        id_to_label[i] = label
    # Add 0917 END

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    config = tf.compat.v1.ConfigProto()
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        session_config=config,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=3,
    )

    # run_config = tf.estimator.RunConfig(
    #     model_dir=FLAGS.output_dir,
    #     session_config=config,
    #     save_summary_steps=10,
    #     log_step_count_steps=100,
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     keep_checkpoint_max=2,
    # )

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # run_config = tf.contrib.tpu.RunConfig(
    #     cluster=tpu_cluster_resolver,
    #     master=FLAGS.master,
    #     model_dir=FLAGS.output_dir,
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     tpu_config=tf.contrib.tpu.TPUConfig(
    #         iterations_per_loop=FLAGS.iterations_per_loop,
    #         num_shards=FLAGS.num_tpu_cores,
    #         per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    num_train_examples = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_examples = len(train_examples)

        if FLAGS.num_train_type == "epoch":
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train)
        elif FLAGS.num_train_type == "step":
            num_train_steps = int(FLAGS.num_train)
        else:
            num_train_steps = int(FLAGS.num_train)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        num_train_examples=num_train_examples,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=None,
    )

    # Add 0917
    early_stopping_hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator=estimator,
        metric_name="eval/f1",
        max_steps_without_increase=10 * FLAGS.save_checkpoints_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=FLAGS.save_checkpoints_steps,
    )
    # Add 0917 END

    if FLAGS.do_train:
        train_file_name = os.path.basename(os.path.splitext(FLAGS.train_file)[0])  # train
        train_file = os.path.join(FLAGS.record_dir, f"{train_file_name}.tf_record")

        # logging.info("### Train TF-Record File: {}".format(train_file))
        # file_based_convert_examples_to_features(
        #     train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, dataset_type="train")

        train_file_exists = os.path.exists(train_file)
        logging.info("### Train file exists: {}; Train file: {}".format(train_file_exists, train_file))
        if not train_file_exists:
            file_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, dataset_type="train")

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size,
        )

        if not FLAGS.do_eval:
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file_name = os.path.basename(os.path.splitext(FLAGS.dev_file)[0])  # dev
        eval_file = os.path.join(FLAGS.record_dir, f"{eval_file_name}.tf_record")

        # logging.info("### Eval TF-Record File: {}".format(eval_file))
        # file_based_convert_examples_to_features(
        #     eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, dataset_type="dev")

        eval_file_exists = os.path.exists(eval_file)
        logging.info("### Eval file exists: {}; Eval file: {}".format(eval_file_exists, eval_file))
        if not eval_file_exists:
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, dataset_type="dev")

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=FLAGS.eval_batch_size,
        )

        if FLAGS.do_train:
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)

            def _f1_higher(best_eval_result, current_eval_result):
                default_key = "eval/f1"
                if not best_eval_result or default_key not in best_eval_result:
                    raise ValueError(
                        'best_eval_result cannot be empty or no eval/f1 is found in it.')

                if not current_eval_result or default_key not in current_eval_result:
                    raise ValueError(
                        'current_eval_result cannot be empty or no eval/f1 is found in it.')

                return best_eval_result[default_key] < current_eval_result[default_key]

            exporter = tf.estimator.BestExporter(
                serving_input_receiver_fn=serving_input_fn,
                compare_fn=_f1_higher,
                exports_to_keep=3
            )

            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                              steps=eval_steps,  # steps=None, evaluate on the entire eval dataset
                                              exporters=exporter,
                                              throttle_secs=20,)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:

        test_type = FLAGS.test_type
        test_path = FLAGS.test_path

        predict_dir = FLAGS.predict_dir
        check_path(predict_dir)
        metrics_file_path = os.path.join(predict_dir, "test_metrics.txt")

        logging.info("***** Running Prediction *****")
        logging.info(f"### TEST {test_type.upper()} PATH: {test_path}")
        logging.info(f"### OUTPUT PREDICT PATH: {predict_dir}")

        def read_checkpoint():
            checkpoint_file = os.path.join(FLAGS.output_dir, "checkpoint")
            if not os.path.exists(checkpoint_file):
                return -1
            start_token = "model_checkpoint_path:"
            with open(checkpoint_file, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    if line.startswith(start_token):
                        checkpoint = line[len(start_token):].strip().strip('"')
                        return checkpoint
            return -1

        def compute_metrics_with_confusion(confusion):
            tp, fp, fn, tn = confusion
            epsilon = 1e-6
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
            return precision, recall, f1, accuracy

        def compute_metrics_with_label(true_label_list, pred_label_list, do_negative_metric=False):
            assert (len(true_label_list) == len(pred_label_list))
            true_label_list = [int(label) for label in true_label_list]
            pred_label_list = [int(label) for label in pred_label_list]
            true_label_list = np.asarray(true_label_list)
            pred_label_list = np.asarray(pred_label_list)
            tp = np.sum(true_label_list * pred_label_list)
            fp = np.sum((1 - true_label_list) * pred_label_list)
            fn = np.sum(true_label_list * (1 - pred_label_list))
            tn = np.sum((1 - true_label_list) * (1 - pred_label_list))
            confusion = (tp, fp, fn, tn)
            precision, recall, f1, accuracy = compute_metrics_with_confusion(confusion)
            binary_metric_result = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
                                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}

            if do_negative_metric:
                negative_confusion = (tn, fn, fp, tp)
                n_precision, n_recall, n_f1, n_accuracy = compute_metrics_with_confusion(negative_confusion)
                negative_metric_result = {"negative_precision": n_precision, "negative_recall": n_recall,
                                          "negative_f1": n_f1, "negative_accuracy": n_accuracy}
                m_precision = (precision + n_precision) / 2
                m_recall = (recall + n_recall) / 2
                m_f1 = (f1 + n_f1) / 2
                m_accuracy = (accuracy + n_accuracy) / 2
                macro_metric_result = {"macro_precision": m_precision, "macro_recall": m_recall,
                                       "macro_f1": m_f1, "macro_accuracy": m_accuracy}
                binary_metric_result.update(negative_metric_result)
                binary_metric_result.update(macro_metric_result)

            return binary_metric_result

        def predict_document(test_fpath, record_fpath, output_fpath, do_negative_metric=False):
            test_examples = processor.get_test_examples(test_fpath)
            num_actual_test_examples = len(test_examples)

            if FLAGS.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on.
                while len(test_examples) % FLAGS.predict_batch_size != 0:
                    test_examples.append(PaddingInputExample())

            # logging.info("### Predict TF-Record File: {}".format(record_fpath))
            # file_based_convert_examples_to_features(test_examples, label_list,
            #                                         FLAGS.max_seq_length, tokenizer,
            #                                         record_fpath, dataset_type="test")

            record_file_exists = os.path.exists(record_fpath)
            logging.info(
                "### Predict file exists: {}; Predict file: {}".format(record_file_exists, record_fpath))

            if not record_file_exists:
                file_based_convert_examples_to_features(test_examples, label_list,
                                                        FLAGS.max_seq_length, tokenizer,
                                                        record_fpath, dataset_type="test")

            logging.info("\n***** Running Prediction *****")
            logging.info("### Test File: {}".format(test_fpath))
            logging.info("  Num examples = %d (%d actual, %d padding)",
                         len(test_examples), num_actual_test_examples,
                         len(test_examples) - num_actual_test_examples)
            logging.info("  Batch size = %d", FLAGS.predict_batch_size)

            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = file_based_input_fn_builder(
                input_file=record_fpath,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder,
                batch_size=FLAGS.predict_batch_size,
            )

            result = estimator.predict(input_fn=predict_input_fn)

            with tf.gfile.GFile(output_fpath, "w") as pred_writer:
                num_written_lines = 0
                logging.info("***** Predict Results *****")

                true_labels = [example.label for example in test_examples]  # str label
                pred_labels = []

                for (i, (example, prediction)) in enumerate(zip(test_examples, result)):
                    predictions = prediction["predictions"]
                    probabilities = prediction["probabilities"]
                    if i >= num_actual_test_examples:
                        break
                    pred_label = id_to_label[predictions]  # map int label to str label
                    pred_labels.append(pred_label)
                    # information = [example.text_a, example.text_b, example.label,
                    #                pred_label] if example.text_b else \
                    #     [example.text_a, example.label, pred_label]
                    information = [example.text_sentence, example.label, pred_label]
                    information = "\t".join(information)
                    output_line = "\t".join(
                        str(class_probability)
                        for class_probability in probabilities) + "\n"
                    pred_writer.write(str(information) + "\t" + output_line)

                    num_written_lines += 1
            assert num_written_lines == num_actual_test_examples

            metric_result = compute_metrics_with_label(true_labels, pred_labels, do_negative_metric=do_negative_metric)
            return metric_result

        metric_checkpoint = read_checkpoint()
        metric_threshold = FLAGS.threshold

        if test_type == "file":
            if tf.gfile.Exists(test_path):
                with tf.gfile.GFile(metrics_file_path, "w") as pred_metrics:
                    test_file_path = test_path
                    # test_dir = os.path.dirname(test_file_path)
                    file_name = os.path.basename(os.path.splitext(test_file_path)[0])  # test_file
                    dir_name = os.path.basename(os.path.dirname(test_file_path))  # test
                    record_file_path = os.path.join(FLAGS.record_dir, "{}_predict.tf_record".format(file_name))
                    output_file_path = os.path.join(predict_dir, "{}_{}_results.tsv".format(dir_name, file_name))

                    metric_result = predict_document(test_file_path, record_file_path, output_file_path,
                                                     do_negative_metric=True)

                    performance = f"### Predict Performance: {output_file_path}\n" \
                                  f"### Checkpoint: {metric_checkpoint}, Threshold: {metric_threshold:.2%}\n" \
                                  f"F1: {metric_result['f1']:.2%}, " \
                                  f"Precision: {metric_result['precision']:.2%}, " \
                                  f"Recall: {metric_result['recall']:.2%}, " \
                                  f"Accuracy: {metric_result['accuracy']:.2%}\n" \
                                  f"Macro-F1: {metric_result['macro_f1']:.2%}, " \
                                  f"Macro-Precision: {metric_result['macro_precision']:.2%}, " \
                                  f"Macro-Recall: {metric_result['macro_recall']:.2%}, " \
                                  f"Macro-Accuracy: {metric_result['macro_accuracy']:.2%}\n" \
                                  f"Confusion: (TP: {metric_result['tp']}, FP: {metric_result['fp']}," \
                                  f"FN: {metric_result['fn']}, TN: {metric_result['tn']})"
                    logging.info(f"\n\n{performance}\n\n")
                    pred_metrics.write(f"{performance}\n")
            else:
                logging.info(f"### WRONG FILE: {test_path} do not exist...")

        elif test_type == "dir":
            if tf.gfile.Exists(test_path) and tf.gfile.IsDirectory(test_path):
                test_file_list = tf.gfile.ListDirectory(test_path)
                test_file_list = [file for file in test_file_list if
                                  os.path.splitext(file)[1] == ".txt"]  # remove .tf_record
                test_file_list = sorted(test_file_list)

                precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
                tp_list, fp_list, fn_list, tn_list = [], [], [], []

                with tf.gfile.GFile(metrics_file_path, "w") as pred_metrics:

                    for test_file in test_file_list:
                        test_file_path = os.path.join(test_path, test_file)
                        file_name = os.path.basename(os.path.splitext(test_file_path)[0])   # test_file
                        dir_name = os.path.basename(os.path.dirname(test_file_path))        # test
                        record_file_path = os.path.join(FLAGS.record_dir, "{}_predict.tf_record".format(file_name))
                        output_file_path = os.path.join(predict_dir, "{}_{}_results.tsv".format(dir_name, file_name))

                        metric_result = predict_document(test_file_path, record_file_path, output_file_path)

                        performance = f"### Predict Performance: {output_file_path}\n" \
                                      f"### Checkpoint: {metric_checkpoint}, Threshold: {metric_threshold:.2%}\n" \
                                      f"F1: {metric_result['f1']:.2%}, " \
                                      f"Precision: {metric_result['precision']:.2%}, " \
                                      f"Recall: {metric_result['recall']:.2%}, " \
                                      f"Accuracy: {metric_result['accuracy']:.2%}\n" \
                                      f"Confusion: (TP: {metric_result['tp']}, FP: {metric_result['fp']}," \
                                      f"FN: {metric_result['fn']}, TN: {metric_result['tn']})"
                        logging.info(f"\n\n{performance}\n\n")
                        pred_metrics.write(f"{performance}\n\n")

                        precision_list.append(metric_result["precision"])
                        recall_list.append(metric_result["recall"])
                        f1_list.append(metric_result["f1"])
                        accuracy_list.append(metric_result["accuracy"])
                        tp_list.append(metric_result["tp"])
                        fp_list.append(metric_result["fp"])
                        fn_list.append(metric_result["fn"])
                        tn_list.append(metric_result["tn"])

                    macro_precision = sum(precision_list) / len(precision_list) if len(precision_list) != 0 else 0
                    macro_recall = sum(recall_list) / len(recall_list) if len(recall_list) != 0 else 0
                    macro_f1 = sum(f1_list) / len(f1_list) if len(f1_list) != 0 else 0
                    macro_accuracy = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) != 0 else 0

                    all_tp = sum(tp_list)
                    all_fp = sum(fp_list)
                    all_fn = sum(fn_list)
                    all_tn = sum(tn_list)
                    all_confusion = (all_tp, all_fp, all_fn, all_tn)
                    micro_precision, micro_recall, micro_f1, micro_accuracy = compute_metrics_with_confusion(all_confusion)

                    performance = f"### All Predict Performance ###\n" \
                                  f"### Checkpoint: {metric_checkpoint}, Threshold: {metric_threshold:.2%}\n" \
                                  f"Micro-F1: {micro_f1:.2%}, Micro-Precision: {micro_precision:.2%}, " \
                                  f"Micro-Recall: {micro_recall:.2%}, Micro-Accuracy: {micro_accuracy:.2%}\n" \
                                  f"All Confusion: (TP: {all_tp}, FP: {all_fp}, FN: {all_fn}, TN: {all_tn})\n" \
                                  f"Macro-F1: {macro_f1:.2%}, Macro-Precision: {macro_precision:.2%}, " \
                                  f"Macro-Recall: {macro_recall:.2%}, Macro-Accuracy: {macro_accuracy:.2%}"
                    logging.info(f"\n\n{performance}\n\n")
                    pred_metrics.write(f"{performance}\n")
            else:
                logging.info(f"### WRONG DIR: {test_path} do not exist...")


    if FLAGS.do_export:
        estimator._export_to_tpu = False
        export_dir = FLAGS.export_dir if FLAGS.export_dir else FLAGS.output_dir
        if not tf.gfile.Exists(export_dir):
            tf.gfile.MakeDirs(export_dir)
        estimator.export_savedmodel(export_dir, serving_input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
