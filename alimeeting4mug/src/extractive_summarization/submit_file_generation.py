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

import json
import sys
from collections import defaultdict
import os
from datasets import DatasetDict
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from utils.config import MS_SDK_TOKEN
from collections import defaultdict


def data_parse_fn(all_examples, task, split):
    all_examples = all_examples["content"]
    label_map = {"1": "B-EOP", "0": "O", 1: "B-EOP", 0: "O"}

    out = []
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
                example["topic_segment_ids"].append({"id": example["sentences"][-1]["id"]})
            if example["topic_segment_ids"][-1]["id"] != example["sentences"][-1]["id"]:
                if example["topic_segment_ids"][-1]["id"] < example["sentences"][-1]["id"]:
                    example["topic_segment_ids"][-1]["id"] = example["sentences"][-1]["id"]

            paragraph_seg_key = "org_segment_id" if "org_segment_id" in example else "paragraph_segment_ids"
            sentences, topic_seg_ids, paragraph_seg_ids = example["sentences"], example["topic_segment_ids"], example[
                paragraph_seg_key]
            sentences = [_["s"] for _ in sentences]
            doc_key_sentence_labels = ["O"] * len(sentences)
            tmp = set()
            if "candidate" not in example:
                example["candidate"] = []
            tmp.update([int(sent_id) for can in example["candidate"] for sent_id in can["key_sentence"]])
            doc_key_sentence_ids = sorted(list(tmp))
            for i in range(len(sentences)):
                if i + 1 in doc_key_sentence_ids:
                    doc_key_sentence_labels[i] = label_map[1]

            multi_ref_doc_key_sentence_labels = []
            if example["candidate"]:
                for can in example["candidate"]:
                    multi_ref_doc_key_sentence_labels.append(["O"] * len(sentences))
                    for sent_id in can["key_sentence"]:
                        multi_ref_doc_key_sentence_labels[-1][int(sent_id) - 1] = label_map[1]
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
                        "label": label,
                        "meeting_key": example["meeting_key"]
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
                "sentences": sentences,
                "labels": labels,
                "multi_labels": multi_ref_doc_key_sentence_labels,
                "label": label,
                "meeting_key": example["meeting_key"]
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
                example["topic_segment_ids"].append({"id": example["sentences"][-1]["id"]})
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
                            "multi_labels": [_[topic_left_index:topic_seg_id] for _ in
                                             multi_ref_topic_key_sentence_labels],
                            "meeting_key": example["meeting_key"],
                            "topic_seg_id": topic_seg_id
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

                multi_ref_topic_key_sentence_labels = [_[topic_left_index:topic_seg_id] for _ in
                                                       multi_ref_topic_key_sentence_labels]

                topic_sentences = sentences[topic_left_index:topic_seg_id]
                topic_left_index = topic_seg_id

                topic_samples.append({
                    "sentences": topic_sentences,
                    "labels": labels,
                    "label": "NONE",
                    "multi_labels": multi_ref_topic_key_sentence_labels,
                    "meeting_key": example["meeting_key"],
                    "topic_seg_id": topic_seg_id
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
    return out_dict


def alimeeting4mug_data_download(task):
    from modelscope.hub.api import HubApi
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.constant import DownloadMode
    api = HubApi()
    api.login(MS_SDK_TOKEN)  # online
    input_config_kwargs = {'delimiter': '\t'}
    data = MsDataset.load(
        'Alimeeting4MUG',
        namespace='modelscope',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        subset_name="default",
        **input_config_kwargs)

    return DatasetDict(
        {
            k: dataset.map(data_parse_fn, batched=True, batch_size=100, remove_columns=["idx", "content"],
                           fn_kwargs={"split": k, "task": task})
            for k, dataset in data
        }
    )


def transfer(input_file_path, output_file_path, task):
    out = []
    raw_datasets = alimeeting4mug_data_download(task)
    test_data = raw_datasets["test"]
    with open(input_file_path, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
        assert len(samples) == len(test_data), "len(samples)={} len(test_data)={}".format(len(samples), len(test_data))
        if task == "topic_key_sentence_extraction":
            out_sample = defaultdict(list)
            for pred_sample, raw_sample in zip(samples, test_data):
                assert len(pred_sample["sentences"]) == len(raw_sample["sentences"]), "len(pred_sample['sentences'])={}, len(raw_sample['sentences'])={}".format(len(pred_sample["sentences"]), len(raw_sample["sentences"]))
                sentences = pred_sample["sentences"]
                meeting_key = raw_sample["meeting_key"]
                predictions = pred_sample["predictions"]
                # print(meeting_key, len(sentences), len(predictions))
                assert len(predictions) == len(sentences)
                topic_seg_id = raw_sample["topic_seg_id"]
                start_sentence_index = topic_seg_id - len(sentences)

                topic_out = []
                for sent_index, pred_label in enumerate(predictions):
                    sent_id = start_sentence_index + sent_index + 1
                    if pred_label != "O":
                        topic_out.append(sent_id)

                out_sample[meeting_key].append({"id":topic_seg_id, "key_sentence":topic_out})

            for key in out_sample.keys():
                out.append(json.dumps({"meeting_key": key, "topic_segment_ids":out_sample[key]}, ensure_ascii=False) + "\n")
        elif task == "doc_key_sentence_extraction":
            for pred_sample, raw_sample in zip(samples, test_data):
                assert len(pred_sample["sentences"]) == len(raw_sample[
                                                                "sentences"]), "len(pred_sample['sentences'])={}, len(raw_sample['sentences'])={}".format(
                    len(pred_sample["sentences"]), len(raw_sample["sentences"]))
                sentences = pred_sample["sentences"]
                meeting_key = raw_sample["meeting_key"]
                predictions = pred_sample["predictions"]
                # print(meeting_key, len(sentences), len(predictions))
                assert len(predictions) == len(sentences)

                es_out = []
                for sent_index, pred_label in enumerate(predictions):
                    sent_id = sent_index + 1
                    if pred_label != "O":
                        es_out.append(sent_id)

                out.append(json.dumps({"meeting_key": meeting_key, "key_sentence": es_out}, ensure_ascii=False) + "\n")

    with open(output_file_path, 'w') as f:
        f.writelines(out)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 4
    task = args[1]
    input_file = args[2]
    output_file = args[3]
    transfer(input_file, output_file, task)
