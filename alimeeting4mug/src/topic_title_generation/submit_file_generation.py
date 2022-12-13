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


def data_parse_fn(all_examples, split, task):
    all_examples = all_examples["content"]
    print("task: {}; split: {}".format(task, split))
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
                if "candidate" not in topic:
                    num_ref = 3
                    topic_titles = [""] * 3
                else:
                    topic_titles = [candidate["title"] for candidate in topic["candidate"]]

                if strategy == "single":
                    labels = topic_titles[0]
                    topic_samples.append({
                        "example_id": example_id,
                        "src_txt": "".join(sentences[topic_left_index:topic_seg_id]),
                        "tgt_txt": labels,
                        "multi_ref": topic_titles,
                        "meeting_key": example["meeting_key"],
                        "topic_seg_id": topic_seg_id
                    })
                elif strategy == "pool":
                    for other_labels in topic_titles:
                        topic_samples.append({
                            "example_id": example_id,
                            "src_txt": "".join(sentences[topic_left_index:topic_seg_id]),
                            "tgt_txt": other_labels,
                            "multi_ref": topic_titles,
                            "meeting_key": example["meeting_key"],
                            "topic_seg_id": topic_seg_id
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
        out_sample = defaultdict(list)

        for pred_sample, raw_sample in zip(samples, test_data):
            meeting_key = raw_sample["meeting_key"]
            predictions = pred_sample["tgt_txt"]
            topic_seg_id = raw_sample["topic_seg_id"]

            out_sample[meeting_key].append({"id":topic_seg_id, "title":predictions})

        for key in out_sample.keys():
            out.append(json.dumps({"meeting_key": key, "topic_segment_ids":out_sample[key]}, ensure_ascii=False) + "\n")

    with open(output_file_path, 'w') as f:
        f.writelines(out)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 3
    input_file = args[1]
    output_file = args[2]
    transfer(input_file, output_file, task="topic_title_generation")
