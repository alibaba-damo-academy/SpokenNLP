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
                    "meeting_key": example["meeting_key"],
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
                           fn_kwargs={"split": k, "task": "action_detection"})
            for k, dataset in data
        }
    )


def transfer(input_file_path, output_file_path):
    out = []
    raw_datasets = alimeeting4mug_data_download()
    test_data = raw_datasets["test"]
    with open(input_file_path, 'r') as f:
        samples = [line.strip().split("\t")[1] for line in f if line.strip()]
        assert len(samples) == len(test_data), "len(samples)={}, len(test_data)={}".format(len(samples), len(test_data))
        out_samples = defaultdict(list)
        for pred_sample, raw_sample in zip(samples, test_data):
            out_samples[raw_sample["meeting_key"]].append(pred_sample)

        for key in out_samples.keys():
            out_sample = {}
            out_sample["meeting_key"] = key
            out_sample["action_ids"] = []
            for sent_index, pred in enumerate(out_samples[key]):
                if pred == "1":
                    out_sample["action_ids"].append({"id": sent_index+1})
            out.append(json.dumps(out_sample, ensure_ascii=False) + "\n")

    with open(output_file_path, 'w') as f:
        f.writelines(out)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 3
    input_file = args[1]
    output_file = args[2]
    transfer(input_file, output_file)
