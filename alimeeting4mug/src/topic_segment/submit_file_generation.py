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

def data_parse_fn(all_examples, task, split):
    all_examples = all_examples["content"]

    out = []
    if task == "topic_segmentation":
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
            out.append({
                "example_id": example_id,
                "meeting_key": example["meeting_key"],
                "sentences": sentences,
                "paragraph_seg_ids": paragraph_seg_ids
            })
    else:
        raise NotImplementedError

    out_dict = defaultdict(list)
    for sample in out:
        for key in sample.keys():
            out_dict[key].append(sample[key])
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
        subset_name="only_topic_segmentation",
        **input_config_kwargs)

    return DatasetDict(
        {
            k: dataset.map(data_parse_fn, batched=True, batch_size=100, remove_columns=["idx", "content"],
                           fn_kwargs={"split": k, "task": "topic_segmentation"})
            for k, dataset in data
        }
    )


def transfer(input_file_path, output_file_path):
    out = []
    raw_datasets = alimeeting4mug_data_download()
    test_data = raw_datasets["test"]
    with open(input_file_path, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
        assert len(samples) == len(test_data)
        for pred_sample, raw_sample in zip(samples, test_data):
            assert len(pred_sample["sentences"]) == len(raw_sample["sentences"])
            sentences = pred_sample["sentences"]
            meeting_key = raw_sample["meeting_key"]
            predictions = pred_sample["predictions"]
            predictions.append("B-EOP")
            paragraph_seg_ids = raw_sample["paragraph_seg_ids"]
            # print(meeting_key, len(sentences), len(predictions), len(paragraph_seg_ids))
            assert len(predictions) == len(paragraph_seg_ids)

            out_sample = {}
            out_sample["meeting_key"] = meeting_key
            out_sample["topic_segment_ids"] = []
            for sent_id, pred_label in zip(paragraph_seg_ids, predictions):
                if pred_label != "O":
                    out_sample["topic_segment_ids"].append(sent_id)
            out.append(json.dumps(out_sample, ensure_ascii=False) + "\n")

    with open(output_file_path, 'w') as f:
        f.writelines(out)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 3
    input_file = args[1]
    output_file = args[2]
    transfer(input_file, output_file)
