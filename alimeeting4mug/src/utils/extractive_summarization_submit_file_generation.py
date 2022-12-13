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
import sys
import json


def merge(topic_es_input_file, doc_es_input_file, output_file):
    with open(topic_es_input_file, 'r') as f:
        topic_es_res = [json.loads(line) for line in f.readlines()]
    with open(doc_es_input_file, 'r') as f:
        doc_es_res = [json.loads(line) for line in f.readlines()]
    assert len(topic_es_res) == len(doc_es_res)
    out = []
    for tes , des in zip(topic_es_res, doc_es_res):
        assert tes["meeting_key"] == des["meeting_key"]
        tes["key_sentence"] = des["key_sentence"]
        out.append(json.dumps(tes, ensure_ascii=False) + "\n")

    with open(output_file, 'w') as f:
        f.writelines(out)


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 4, "python extractive_summarization_submit_file_generation.py topic_es_submit.json doc_es_submit.json es_submit.json"
    topic_es_input_fp = args[1]
    doc_es_input_fp = args[2]
    output_fp = args[3]
    merge(topic_es_input_fp, doc_es_input_fp, output_fp)
