# coding=utf-8
# Copyright (c) 2023, Alibaba Group.
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

import re
import os
import json
import codecs
import random
import numpy as np
import xml.dom.minidom as xmldom
from tqdm import tqdm
from collections import defaultdict


AMI_DIR = "../data/AMI/ami_public_manual_1.6.2"
DATASET_DIR = "../data/AMI/dataset"
WORD_DIR = os.path.join(AMI_DIR, "words")
DACT_DIR = os.path.join(AMI_DIR, "dialogueActs")
ABST_DIR = os.path.join(AMI_DIR, "abstractive")
LINK_DIR = os.path.join(AMI_DIR, "extractive")
ONTO_DIR = os.path.join(AMI_DIR, "ontologies")
SIMILARITY_PATH = "../data/AMI/similarity/similarity_ngram_cosine_ami101a.json"


def get_default_meet2speakers():
    meet2speakers = defaultdict(list)
    files = os.listdir(WORD_DIR)
    for file_name in files:
        if file_name.startswith("."):
            continue
        file_name_list = file_name.strip().split(".")
        meeting_name = file_name_list[0]
        speaker_name = file_name_list[1]
        meet2speakers[meeting_name].append(speaker_name)
    meet2speakers = dict(sorted(meet2speakers.items(), key=lambda x: x[0]))
    meet2speakers = {meet: sorted(speakers) for meet, speakers in meet2speakers.items()}
    return meet2speakers


def test_get_default_meet2speakers():
    meet2speakers = get_default_meet2speakers()
    print(meet2speakers)
    return meet2speakers


def parse_abstractive(abstractive_path):
    ab_dobj = xmldom.parse(abstractive_path)
    ab_eobj = ab_dobj.documentElement

    summary_list = ab_eobj.getElementsByTagName("abstract")
    action_list = ab_eobj.getElementsByTagName("actions")
    decision_list = ab_eobj.getElementsByTagName("decisions")
    problem_list = ab_eobj.getElementsByTagName("problems")

    def get_special_dict(special_list):
        special_dict = {}  # key: id, value: sentence
        for special_info in special_list:
            if special_info.hasChildNodes:
                for special_line in special_info.childNodes:
                    if type(special_line) == xmldom.Element:
                        special_id = special_line.getAttribute("nite:id")
                        special_sentence = special_line.firstChild.data
                        special_dict[special_id] = special_sentence
        return special_dict

    summary_dict = get_special_dict(summary_list)
    action_dict = get_special_dict(action_list)
    decision_dict = get_special_dict(decision_list)
    problem_dict = get_special_dict(problem_list)

    abstract_dict = {"abstract": summary_dict, "action": action_dict, "decision": decision_dict,
                     "problem": problem_dict}
    return abstract_dict


def test_parse_abstractive(meet="ES2013b"):
    abstractive_file = os.path.join(ABST_DIR, f"{meet}.abssumm.xml")
    abstract_dict = parse_abstractive(abstractive_file)
    print(abstract_dict)
    return abstract_dict


def parse_extractive(link_path):
    link_dobj = xmldom.parse(link_path)
    link_eobj = link_dobj.documentElement
    link_list = link_eobj.getElementsByTagName("summlink")

    def get_special_id(node, role):
        assert node.getAttribute("role") == role
        href = node.getAttribute("href")
        href = href.strip().split("#")[-1]
        special_id = re.findall(r"^id\((.*)\)", href)[0]
        return special_id

    link_dict = defaultdict(list)  # key: dialog_act_id, value: abstract_id list
    for link_info in link_list:
        if link_info.hasChildNodes:
            extract_node = link_info.childNodes[1]
            dialog_act_id = get_special_id(extract_node, "extractive")
            abstract_node = link_info.childNodes[3]
            abstract_id = get_special_id(abstract_node, "abstractive")
            link_dict[dialog_act_id].append(abstract_id)
    return link_dict


def test_parse_extractive(meet="ES2013b"):
    link_file = os.path.join(LINK_DIR, f"{meet}.summlink.xml")
    extract_dict = parse_extractive(link_file)
    print(extract_dict)
    return extract_dict


def get_default_dialog_act_type_dict():
    da_type_file = os.path.join(ONTO_DIR, "da-types.xml")
    datype_dobj = xmldom.parse(da_type_file)
    datype_eobj = datype_dobj.documentElement
    datype_list = datype_eobj.getElementsByTagName("da-type")
    da_dict = {}  # key: da_type_id, value: da_type_name
    for da_class in datype_list:
        if da_class.hasChildNodes():
            class_name = da_class.getAttribute("gloss")
            for da_type in da_class.childNodes:
                if type(da_type) == xmldom.Element:
                    type_id = da_type.getAttribute("nite:id")
                    type_name = da_type.getAttribute("gloss")
                    type_name = "{}#{}".format(class_name, type_name)
                    da_dict[type_id] = type_name
    return da_dict


def test_get_default_dialog_act_type_dict():
    dialog_act_type_dict = get_default_dialog_act_type_dict()
    print(dialog_act_type_dict)
    return dialog_act_type_dict


def parse_dialogue_act(dialogue_act_path):
    da_dobj = xmldom.parse(dialogue_act_path)
    da_eobj = da_dobj.documentElement

    da_dict = {}  # key: id, value: {dialog_act, start_id, end_id}
    da_type_dict = get_default_dialog_act_type_dict()

    file_names = os.path.basename(dialogue_act_path).strip().split(".")
    meeting_name = file_names[0]
    speaker_name = file_names[1]
    data_source = f"AMI#{meeting_name[:2]}"

    da_list = da_eobj.getElementsByTagName("dact")
    for da_info in da_list:
        da_id = da_info.getAttribute("nite:id")

        type_node_list = da_info.getElementsByTagName("nite:pointer")
        if len(type_node_list) == 1:
            type_node = type_node_list[0]
            da_type = type_node.getAttribute("href")
            da_type = re.findall(r"da\-types\.xml#id\((.*)\)", da_type)[0]
            da_type = da_type_dict.get(da_type, "Unlab#Unlab")
        else:
            da_type = "Unlab#Unlab"

        word_node_list = da_info.getElementsByTagName("nite:child")
        assert len(word_node_list) == 1
        word_node = word_node_list[0]
        word_id = word_node.getAttribute("href")
        word_id = word_id.split("#")[-1].split("..")
        word_id = [int(re.findall(r"id\(.*words(\d*)\)", x)[0]) for x in word_id]
        word_id = [word_id[0], word_id[0]] if len(word_id) == 1 else word_id
        assert len(word_id) == 2
        start_word_id, end_word_id = word_id

        da_data = {"dact_types": da_type, "start_id": start_word_id, "end_id": end_word_id, "dact_ids": da_id,
                   "meeting_name": meeting_name, "speaker_name": speaker_name, "data_source": data_source}
        da_dict[da_id] = da_data

    da_dict = dict(sorted(da_dict.items(), key=lambda x: (x[1]["start_id"], x[1]["end_id"])))

    return da_dict


def test_parse_dialogue_act(meet="ES2015b", speaker="D"):
    da_file = os.path.join(DACT_DIR, f"{meet}.{speaker}.dialog-act.xml")
    da_dict = parse_dialogue_act(da_file)
    print(da_dict)
    return da_dict


def get_extend_time_dict():
    time_dict = {"ES2008a.A.words.xml#1315": {"start_time": "1041.15", "end_time": "1041.30"},
                 "ES2008a.A.words.xml#1316": {"start_time": "1041.30", "end_time": "1041.45"},
                 "ES2008a.A.words.xml#1317": {"start_time": "1041.45", "end_time": "1042.00"},
                 "TS3006d.A.words.xml#2182": {"start_time": "2552.767", "end_time": "2552.767"},
                 "TS3006d.A.words.xml#2521": {"start_time": "2921.133", "end_time": "2921.133"},
                 "TS3006d.B.words.xml#1435": {"start_time": "2553.424", "end_time": "2553.424"},
                 "TS3006d.D.words.xml#2700": {"start_time": "1646.72", "end_time": "1646.72"},
                 "TS3006d.D.words.xml#4318": {"start_time": "2904.125", "end_time": "2904.125"},
                 "TS3006d.D.words.xml#4319": {"start_time": "2904.126", "end_time": "2904.126"},
                 "TS3007c.D.words.xml#45": {"start_time": "204.57", "end_time": "204.57"},
                 "TS3010b.A.words.xml#920": {"start_time": "1703.713", "end_time": "1703.713"},
                 "TS3010b.D.words.xml#1156": {"start_time": "1702.337", "end_time": "1702.337"},
                 "TS3011a.A.words.xml#36": {"start_time": "37.344", "end_time": "37.344"},
                 "TS3011d.A.words.xml#2271": {"start_time": "2102.585", "end_time": "2102.585"},
                 "TS3011d.B.words.xml#580": {"start_time": "1235.44", "end_time": "1235.44"},
                 "TS3011d.B.words.xml#1139": {"start_time": "2142.657", "end_time": "2142.657"},
                 "TS3011d.B.words.xml#1140": {"start_time": "2142.658", "end_time": "2142.658"},
                 "EN2001d.C.words.xml#1": {"start_time": "47.473", "end_time": "48.097"},
                 "EN2002d.D.words.xml#1054": {"start_time": "756.993", "end_time": "756.993"},
                 "EN2006a.C.words.xml#392": {"start_time": "235.824", "end_time": "235.824"},
                 "IB4002.C.words.xml#1984": {"start_time": "1854.989", "end_time": "1854.989"},
                 "IN1007.D.words.xml#344": {"start_time": "316.849", "end_time": "316.849"}}
    return time_dict


def parse_words(words_path):
    words_dobj = xmldom.parse(words_path)
    words_eobj = words_dobj.documentElement
    words_dict = {}  # key: word_id, value: word_data
    time_dict = get_extend_time_dict()

    def get_word_data(word_node, word_type="w"):
        word_id = word_node.getAttribute("nite:id")
        word_id = word_id.strip().split(".")[-1]
        word_id = re.findall(r"words(\d+)", word_id)
        if len(word_id) != 1:
            return {}
        word_id = int(word_id[0])

        if word_type == "w":
            word = word_node.firstChild.data
        elif word_type == "disfmarker":
            word = "..."
        else:
            word = ""

        if word_node.hasAttribute("starttime"):
            start_time = word_node.getAttribute("starttime")
            if word_node.hasAttribute("endtime"):
                end_time = word_node.getAttribute("endtime")
            else:
                end_time = start_time
        else:
            line_id = f"{os.path.basename(words_path)}#{word_id}"
            start_time = time_dict[line_id]["start_time"]
            end_time = time_dict[line_id]["end_time"]

        start_time = float(start_time)
        end_time = float(end_time)
        word_data = {"word": word, "word_id": word_id, "word_type": word_type,
                     "start_time": start_time, "end_time": end_time}
        return word_data

    word_type_list = ["w", "disfmarker", "gap", "vocalsound", "transformerror"]
    for input_word_type in word_type_list:
        word_node_list = words_eobj.getElementsByTagName(input_word_type)
        for input_word_node in word_node_list:
            input_word_data = get_word_data(input_word_node, word_type=input_word_type)
            if input_word_data:
                words_dict[input_word_data["word_id"]] = input_word_data

    return words_dict


def test_parse_words(meet="ES2013b", speaker="D"):
    word_file = os.path.join(WORD_DIR, f"{meet}.{speaker}.words.xml")
    word_dict = parse_words(word_file)
    print(word_dict)
    return word_dict


def dialogue_act_link_words(dialogue_act, words):
    dact_dict = {}
    if isinstance(dialogue_act, str) and os.path.exists(dialogue_act):
        dact_dict = parse_dialogue_act(dialogue_act)
    elif isinstance(dialogue_act, dict):
        dact_dict = dialogue_act

    words_dict = {}
    if isinstance(words, str) and os.path.exists(words):
        words_dict = parse_words(words)
    elif isinstance(words, dict):
        words_dict = words

    for dact_id, dact_data in dact_dict.items():
        start_id = dact_data["start_id"]
        end_id = dact_data["end_id"]

        word_list, start_time_list, end_time_list = [], [], []
        for word_id in range(start_id, end_id + 1):
            if word_id not in words_dict:
                continue
            word_data = words_dict[word_id]
            word_list.append(word_data["word"])
            start_time_list.append(word_data["start_time"])
            end_time_list.append(word_data["end_time"])

        sentence = " ".join(word_list)
        start_time = start_time_list[0]
        end_time = end_time_list[-1]

        dact_data["sentence"] = sentence
        dact_data["start_time"] = start_time
        dact_data["end_time"] = end_time

    return dact_dict


def test_dialogue_act_link_words(meet="ES2013b", speaker="D"):
    words_file = os.path.join(WORD_DIR, f"{meet}.{speaker}.words.xml")
    dialogue_act_file = os.path.join(DACT_DIR, f"{meet}.{speaker}.dialog-act.xml")

    # Test 1
    dact_dict = dialogue_act_link_words(dialogue_act_file, words_file)

    # Test 2
    dact_dict = parse_dialogue_act(dialogue_act_file)
    dact_dict = dialogue_act_link_words(dact_dict, words_file)

    # Test 3
    words_dict = parse_words(words_file)
    dact_dict = dialogue_act_link_words(dialogue_act_file, words_dict)

    # Test 4
    words_dict = parse_words(words_file)
    dact_dict = parse_dialogue_act(dialogue_act_file)
    dact_dict = dialogue_act_link_words(dact_dict, words_dict)

    print(dact_dict)

    return dact_dict


def dialogue_act_link_label(dialogue_act, extractive, abstractive):
    dact_dict = {}
    if isinstance(dialogue_act, str) and os.path.exists(dialogue_act):
        dact_dict = parse_dialogue_act(dialogue_act)
    elif isinstance(dialogue_act, dict):
        dact_dict = dialogue_act

    extract_dict = {}
    if isinstance(extractive, str) and os.path.exists(extractive):
        extract_dict = parse_extractive(extractive)
    elif isinstance(extractive, dict):
        extract_dict = extractive

    abstract_dict = {}
    if isinstance(abstractive, str) and os.path.exists(abstractive):
        abstract_dict = parse_abstractive(abstractive)
    elif isinstance(abstractive, dict):
        abstract_dict = abstractive
    abstract_action_dict = abstract_dict["action"]

    for dact_id, dact_data in dact_dict.items():
        dact_data["action_label"] = 0
        dact_data["action_description"] = ""
        if dact_id not in extract_dict:
            continue
        abstract_ids = extract_dict[dact_id]
        for abstract_id in abstract_ids:
            if abstract_id in abstract_action_dict:
                action_label = 1
                action_description = abstract_action_dict[abstract_id]
                dact_data["action_label"] = action_label
                dact_data["action_description"] = action_description
                break

    return dact_dict


def test_dialogue_act_link_label(meet="ES2013b", speaker="D"):
    dialogue_act_file = os.path.join(DACT_DIR, f"{meet}.{speaker}.dialog-act.xml")
    extractive_file = os.path.join(LINK_DIR, f"{meet}.summlink.xml")
    abstractive_file = os.path.join(ABST_DIR, f"{meet}.abssumm.xml")
    dact_dict = dialogue_act_link_label(dialogue_act_file, extractive_file, abstractive_file)
    print(dact_dict)
    return dact_dict


def is_valid_meeting(meeting_name, remove_zero=True):
    abstractive_file = os.path.join(ABST_DIR, f"{meeting_name}.abssumm.xml")
    abstract_dict = parse_abstractive(abstractive_file)
    if len(abstract_dict["action"]) == 0:
        if remove_zero:
            return False
        return True

    link_file = os.path.join(LINK_DIR, f"{meeting_name}.summlink.xml")
    extract_dict = parse_extractive(link_file)

    extract_abstract_ids = []
    for dialog_act_id, abstract_ids in extract_dict.items():
        for abstract_id in abstract_ids:
            if abstract_id not in extract_abstract_ids:
                extract_abstract_ids.append(abstract_id)

    abstract_action_ids = []
    for abstract_id, abstract_sentence in abstract_dict["action"].items():
        if abstract_id not in abstract_action_ids:
            abstract_action_ids.append(abstract_id)

    candidate_ids = [idx for idx in abstract_action_ids if idx in extract_abstract_ids]
    count_action_sentence = len(candidate_ids)
    if count_action_sentence >= 1:
        return True
    return False


def default_sort_ami_data_list(ami_data_list):
    meet_types = ["IS", "ES", "TS", "IB", "EN", "IN"]
    meet_types = {f"AMI#{meet_type}": idx for idx, meet_type in enumerate(meet_types)}
    ami_data_list = sorted(ami_data_list, key=lambda x: (meet_types[x["data_source"]], x["meeting_name"], x["sentence_id"]))
    for idx, data_item in enumerate(ami_data_list):
        data_item["line_id"] = idx
    return ami_data_list


def default_sort_meet_data_list(meet_data_list):
    meet_data_list = sorted(meet_data_list, key=lambda x: (x["start_time"], x["end_time"]))
    for idx, data_item in enumerate(meet_data_list):
        data_item["sentence_id"] = idx + 1
    return meet_data_list


def convert_meet_dict_to_list(data_dict):
    data_list = []
    for data_id, data in data_dict.items():
        data_item = {}
        data_item["sentence"] = data.get("sentence", "").strip()
        if data_item["sentence"] == "":
            continue
        data_item["dact_ids"] = data.get("dact_ids", "")
        data_item["dact_types"] = data.get("dact_types", "")
        data_item["action_label"] = data.get("action_label", 0)
        data_item["action_description"] = data.get("action_description", "")
        data_item["meeting_name"] = data.get("meeting_name", "")
        data_item["speaker_name"] = data.get("speaker_name", "")
        data_item["start_time"] = data.get("start_time", 0.0)
        data_item["end_time"] = data.get("end_time", 0.0)
        data_item["data_source"] = data.get("data_source", "")
        data_item["sentence"] = data.get("sentence", "")
        data_list.append(data_item)
    meet_data_list = default_sort_meet_data_list(data_list)
    return meet_data_list


def status_ami_data_list(ami_data_list, detail_type=False, detail_file=False, detail_document=False):
    meeting_dict = {}  # key: meeting_name, value: meeting_status

    for data_item in ami_data_list:
        meeting_name = data_item["meeting_name"]
        action_label = data_item["action_label"]
        if meeting_name not in meeting_dict:
            meeting_dict[meeting_name] = defaultdict(int)
        meeting_dict[meeting_name]["num_sentence"] += 1
        if action_label == 1:
            meeting_dict[meeting_name]["num_positive"] += 1
        else:
            meeting_dict[meeting_name]["num_negative"] += 1

    count_meeting = len(meeting_dict)
    count_es, count_ts, count_is = 0, 0, 0
    # count_en, count_in, count_ib = 0, 0, 0
    count_sentence, count_positive, count_negative = 0, 0, 0
    for meeting_name, meeting_status in meeting_dict.items():
        count_es = count_es + 1 if meeting_name.startswith("ES") else count_es
        count_ts = count_ts + 1 if meeting_name.startswith("TS") else count_ts
        count_is = count_is + 1 if meeting_name.startswith("IS") else count_is
        # count_ib = count_ib + 1 if meeting_name.startswith("IB") else count_ib
        # count_en = count_en + 1 if meeting_name.startswith("EN") else count_en
        # count_in = count_in + 1 if meeting_name.startswith("IN") else count_in
        count_sentence += meeting_status["num_sentence"]
        count_positive += meeting_status["num_positive"]
        count_negative += meeting_status["num_negative"]

    rate_positive = count_positive / (count_sentence + 1e-6)
    rate_negative = count_negative / (count_sentence + 1e-6)

    print(f"## Overall Status:")
    print(f"## Meeting: {count_meeting} (IS: {count_is}, ES: {count_es}, TS: {count_ts})")
    print(f"## All Sentence: {count_sentence}, Positive: {count_positive} ({rate_positive:.2%}), "
          f"Negative: {count_negative} ({rate_negative:.2%})\n")

    if detail_document:
        count_sentence_list = [status_item["num_sentence"] for meeting_name, status_item in meeting_dict.items()]
        count_positive_list = [status_item["num_positive"] for meeting_name, status_item in meeting_dict.items()]
        count_negative_list = [status_item["num_negative"] for meeting_name, status_item in meeting_dict.items()]

        count_positive_list = np.asarray(count_positive_list)
        avg_positive = np.mean(count_positive_list)
        std_positive = np.std(count_positive_list)
        p25_positive = np.percentile(count_positive_list, 25)
        p50_positive = np.percentile(count_positive_list, 50)
        p75_positive = np.percentile(count_positive_list, 75)
        min_positive = np.min(count_positive_list)
        max_positive = np.max(count_positive_list)

        print(f"# Avg Positive: {avg_positive:.2f}")
        print(f"# Std Positive: {std_positive:.2f}")
        print(f"# P25 Positive: {p25_positive:.2f}")
        print(f"# P50 Positive: {p50_positive:.2f}")
        print(f"# P75 Positive: {p75_positive:.2f}")
        print(f"# Min Positive: {min_positive:.2f}")
        print(f"# Max Positive: {max_positive:.2f}")

    if detail_type:
        meet_types = ["IS", "ES", "TS"]
        print(f"\n## Meeting Type Detail Status:")
        for meet_type in meet_types:
            count_meeting = 0
            count_sentence, count_positive, count_negative = 0, 0, 0
            for meeting_name, meeting_status in meeting_dict.items():
                if not meeting_name.startswith(meet_type):
                    continue
                count_meeting += 1
                count_sentence += meeting_status["num_sentence"]
                count_positive += meeting_status["num_positive"]
                count_negative += meeting_status["num_negative"]

            rate_positive = count_positive / (count_sentence + 1e-6)
            rate_negative = count_negative / (count_sentence + 1e-6)

            print(f"## {meet_type} Meeting: {count_meeting}")
            print(f"## All Sentence: {count_sentence}, Positive: {count_positive} ({rate_positive:.2%}), "
                  f"Negative: {count_negative} ({rate_negative:.2%})\n")

    if detail_file:
        print("\n## Meeting File Detail Status:")
        for meeting_idx, (meeting_name, meeting_status) in enumerate(meeting_dict.items()):
            count_sentence = meeting_status["num_sentence"]
            count_positive = meeting_status["num_positive"]
            count_negative = meeting_status["num_negative"]

            rate_positive = count_positive / (count_sentence + 1e-6)
            rate_negative = count_negative / (count_sentence + 1e-6)

            print(f"## File {meeting_idx + 1:03d}: [{meeting_name}] All Sentence: {count_sentence}, "
                  f"Positive: {count_positive} ({rate_positive:.2%}), "
                  f"Negative: {count_negative} ({rate_negative:.2%}).")


def scenario_only_dataset_split(ami_data_list):

    def which_dataset(meet_name):
        training = ("ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010", "ES2012", "ES2013",
                    "ES2015", "ES2016", "IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006",
                    "IS1007", "TS3005", "TS3008", "TS3009", "TS3010", "TS3011", "TS3012")
        development = ("ES2003", "ES2011", "IS1008", "TS3004", "TS3006")
        prediction = ("ES2004", "ES2014", "IS1009", "TS3003", "TS3007")
        meet_name = meet_name[:6]
        if meet_name in training:
            return "train"
        elif meet_name in development:
            return "dev"
        elif meet_name in prediction:
            return "test"
        return "none"

    for data_item in ami_data_list:
        meeting_name = data_item["meeting_name"]
        dataset = which_dataset(meeting_name)
        assert dataset in ["train", "dev", "test"]
        data_item["dataset"] = dataset

    train_data_list = [item for item in ami_data_list if item["dataset"] == "train"]
    print("\nTraining Dataset:")
    status_ami_data_list(train_data_list, detail_document=True)

    dev_data_list = [item for item in ami_data_list if item["dataset"] == "dev"]
    print("\nDevelopment Dataset:")
    status_ami_data_list(dev_data_list, detail_document=True)

    test_data_list = [item for item in ami_data_list if item["dataset"] == "test"]
    print("\nTest Dataset:")
    status_ami_data_list(test_data_list, detail_document=True)

    return ami_data_list


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def single_select_ami_data_list(data_list, select_key, select_values):
    result_list = [item for item in data_list if item[select_key] in select_values]
    return result_list


def select_ami_data_list(data_list, select_dict):
    result_list = data_list
    for select_key, select_values in select_dict.items():
        result_list = single_select_ami_data_list(result_list, select_key, select_values)
    return result_list


def load_similarity(file_path):
    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        similarity_map = json.load(fr)
        return similarity_map


def add_new_field(result_data_list, ami_data_list, new_field_list=None,
                  num_left_context=1, num_right_context=1, num_global_context=1,
                  add_context_label=True, context_sep="###", context_label_sep="@@@"):
    if not new_field_list:
        return result_data_list

    dataset_dict = defaultdict(dict)  # key: meeting_name, value: key: sentence_id, value: sentence
    for data_item in ami_data_list:
        meeting_name = data_item["meeting_name"]
        sentence_id = data_item["sentence_id"]
        sentence = data_item["sentence"]
        label = data_item["action_label"]
        item = {"sentence": str(sentence), "label": int(label)}
        dataset_dict[meeting_name][sentence_id] = item

    similarity_map = {}
    if "global_context" in new_field_list:
        similarity_file = SIMILARITY_PATH
        similarity_map = load_similarity(similarity_file)

    for data_item in tqdm(result_data_list, desc="Add Fields: "):
        current_meeting_name = data_item["meeting_name"]
        current_sentence_id = data_item["sentence_id"]

        new_field = "global_context"
        if new_field in new_field_list:
            global_context_list = []
            similarity_list = similarity_map[current_meeting_name].get(str(current_sentence_id), None)
            if similarity_list is not None:
                for global_idx in range(num_global_context):
                    score = similarity_list[global_idx]["score"]
                    if score == 0.0:
                        continue
                    global_sentence_id = similarity_list[global_idx]["idx"]
                    global_item = dataset_dict[current_meeting_name].get(global_sentence_id, None)
                    if global_item is not None:
                        global_sentence = global_item["sentence"]
                        global_label = global_item["label"]
                        global_span = f"{global_sentence}"
                        if add_context_label:
                            global_span = f"{global_sentence}{context_label_sep}{global_label}"
                        global_context_list.append(global_span)
            global_context = context_sep.join(global_context_list) if len(global_context_list) > 0 else context_sep
            data_item[new_field] = global_context

        new_field = "left_context"
        if new_field in new_field_list:
            left_context_list = []
            for left_sentence_id in range(current_sentence_id - 1, current_sentence_id - num_left_context - 1, -1):
                if current_meeting_name not in dataset_dict:
                    continue
                left_item = dataset_dict[current_meeting_name].get(left_sentence_id, None)
                if left_item is not None:
                    left_sentence = left_item["sentence"]
                    left_label = left_item["label"]
                    left_span = f"{left_sentence}"
                    if add_context_label:
                        left_span = f"{left_sentence}{context_label_sep}{left_label}"
                    left_context_list.append(left_span)
            left_context = context_sep.join(left_context_list) if len(left_context_list) > 0 else context_sep
            data_item[new_field] = left_context

        new_field = "right_context"
        if new_field in new_field_list:
            right_context_list = []
            for right_sentence_id in range(current_sentence_id + 1, current_sentence_id + num_right_context + 1):
                if current_meeting_name not in dataset_dict:
                    continue
                right_item = dataset_dict[current_meeting_name].get(right_sentence_id, None)
                if right_item is not None:
                    right_sentence = right_item["sentence"]
                    right_label = right_item["label"]
                    right_span = f"{right_sentence}"
                    if add_context_label:
                        right_span = f"{right_sentence}{context_label_sep}{right_label}"
                    right_context_list.append(right_span)
            right_context = context_sep.join(right_context_list) if len(right_context_list) > 0 else context_sep
            data_item[new_field] = right_context

        new_field = "document_length"
        if new_field in new_field_list:
            document_length = len(dataset_dict[current_meeting_name])
            data_item[new_field] = document_length

    return result_data_list


def balance_data_list(data_list):
    positive_data_list, negative_data_list = [], []
    for data_item in data_list:
        label = data_item["action_label"]
        assert label in [0, 1]
        if label == 1:
            positive_data_list.append(data_item)
        else:
            negative_data_list.append(data_item)
    max_data_list, min_data_list = negative_data_list, positive_data_list
    if len(positive_data_list) > len(negative_data_list):
        max_data_list, min_data_list = positive_data_list, negative_data_list
    assert (len(max_data_list) >= len(min_data_list))
    balance_times = int(len(max_data_list) / len(min_data_list)) if len(min_data_list) != 0 else 0
    new_data_list = []
    last_j = 0
    for i in range(len(min_data_list)):
        min_item = min_data_list[i]
        new_data_list.append(min_item)
        for j in range(i * balance_times, (i + 1) * balance_times):
            last_j = j
            if j >= len(max_data_list):
                continue
            max_item = max_data_list[j]
            new_data_list.append(max_item)
    for j in range(last_j + 1, len(max_data_list)):
        max_item = max_data_list[j]
        new_data_list.append(max_item)
    return new_data_list


def write_document(meet_lines, file_path, field_list=None, default_value="###"):
    default_field_list = ["sentence", "label", "line_id"]
    field_list = field_list if field_list else default_field_list

    with codecs.open(file_path, "w", encoding="utf-8") as fw:
        for meet_line in meet_lines:
            output_line = []
            for field in field_list:
                value = str(meet_line[field]).strip()
                value = value if value else default_value
                # print(f"{field}: {value}")
                output_line.append(value)
            output_line = "\t".join(output_line)
            fw.write(f"{output_line}\n")


def output_ami_document(data_list, condition, file_path, field_list=None,
                        num_context_left=1, num_context_right=1, num_global_context=1,
                        add_context_label=True, context_sep="###", context_label_sep="@@@",
                        do_balance=False, do_shuffle=False, seed=2021):
    default_field_list = ["sentence", "action_label", "line_id"]
    field_list = field_list if field_list else default_field_list
    result_data_list = select_ami_data_list(data_list, condition)
    status_ami_data_list(result_data_list)

    meetline_fields = ["sentence", "line_id", "action_label", "action_description", "dact_ids", "dact_types", "dataset",
                       "sentence_id", "meeting_name", "speaker_name", "start_time", "end_time", "data_source"]
    new_field_list = [field for field in field_list if field not in meetline_fields]
    result_data_list = add_new_field(result_data_list, data_list, new_field_list,
                                     num_left_context=num_context_left, num_right_context=num_context_right,
                                     num_global_context=num_global_context, add_context_label=add_context_label,
                                     context_sep=context_sep, context_label_sep=context_label_sep)

    if do_balance:
        result_data_list = balance_data_list(result_data_list)
    if do_shuffle:
        random.seed(seed)
        random.shuffle(result_data_list)
    write_document(result_data_list, file_path, field_list=field_list)


def output_ami_dataset(data_list, dataset_path, supply_fields=None,
                       num_context_left=1, num_context_right=1, num_context_global=1,
                       add_context_label=True, context_sep="###", context_label_sep="@@@", seed=2021):
    check_path(dataset_path)

    # Train
    print("\n### Train: ")
    file_name = "train.txt"
    file = os.path.join(dataset_path, file_name)
    condition = {"dataset": ["train"]}
    output_ami_document(data_list, condition, file, field_list=supply_fields,
                        num_context_left=num_context_left, num_context_right=num_context_right,
                        num_global_context=num_context_global, add_context_label=add_context_label,
                        context_sep=context_sep, context_label_sep=context_label_sep,
                        do_balance=True, do_shuffle=True, seed=seed)

    # Dev
    print("\n### Dev: ")
    file_name = "dev.txt"
    file = os.path.join(dataset_path, file_name)
    condition = {"dataset": ["dev"]}
    output_ami_document(data_list, condition, file, field_list=supply_fields,
                        num_context_left=num_context_left, num_context_right=num_context_right,
                        num_global_context=num_context_global, add_context_label=add_context_label,
                        context_sep=context_sep, context_label_sep=context_label_sep, seed=seed)

    # Test
    print("\n### Test: ")
    file_name = "test.txt"
    file = os.path.join(dataset_path, file_name)
    condition = {"dataset": ["test"]}
    output_ami_document(data_list, condition, file, field_list=supply_fields,
                        num_context_left=num_context_left, num_context_right=num_context_right,
                        num_global_context=num_context_global, add_context_label=add_context_label,
                        context_sep=context_sep, context_label_sep=context_label_sep, seed=seed)


def ami_process_dialogue_act():
    meet2speakers = get_default_meet2speakers()

    ami_data_list = []
    for meet, speakers in tqdm(meet2speakers.items()):
        abstract_file = os.path.join(ABST_DIR, f"{meet}.abssumm.xml")
        extract_file = os.path.join(LINK_DIR, f"{meet}.summlink.xml")
        if not os.path.exists(abstract_file) or not os.path.exists(extract_file):
            continue

        meet_dact_dict = {}
        for speaker in speakers:
            words_file = os.path.join(WORD_DIR, f"{meet}.{speaker}.words.xml")
            dialog_act_file = os.path.join(DACT_DIR, f"{meet}.{speaker}.dialog-act.xml")
            if not os.path.exists(words_file) or not os.path.exists(dialog_act_file):
                continue
            if not is_valid_meeting(meet, remove_zero=True):
                continue
            speaker_dact_dict = dialogue_act_link_words(dialog_act_file, words_file)
            meet_dact_dict.update(speaker_dact_dict)

        meet_dact_dict = dialogue_act_link_label(meet_dact_dict, extract_file, abstract_file)
        meet_data_list = convert_meet_dict_to_list(meet_dact_dict)
        ami_data_list.extend(meet_data_list)

    ami_data_list = default_sort_ami_data_list(ami_data_list)
    status_ami_data_list(ami_data_list, detail_type=True, detail_file=True, detail_document=True)
    ami_data_list = scenario_only_dataset_split(ami_data_list)

    dataset_path = DATASET_DIR
    fields = ["sentence", "action_label", "line_id", "sentence_id", "document_length",
              "left_context", "right_context", "global_context"]
    output_ami_dataset(ami_data_list, dataset_path, supply_fields=fields,
                       num_context_left=2, num_context_right=2, num_context_global=2)


if __name__ == '__main__':
    # test_get_default_meet2speakers()
    # test_parse_abstractive()
    # test_parse_extractive()
    # test_get_default_dialog_act_type_dict()
    # test_parse_dialogue_act()
    # test_parse_words()
    # test_dialogue_act_link_words()
    # test_dialogue_act_link_label()
    ami_process_dialogue_act()

