
import os
import json
import configparser
import argparse

from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from analysis import tokenizer


bTokenizer = tokenizer.BasicTokenizer()
tokenize_func = bTokenizer.tokenize

sec_flag = "========"


def tokenize_method(sec_text):
    # get paragraphs
    sec_paragraphs = list(filter(lambda x: x != '', sec_text.split("\n")))
    # tokenized to sentences by nltk
    sec_sents = [sent_tokenize(p) for p in sec_paragraphs]
    sec_sent_labels = [[-100] * (len(p_sents) - 1) + [0] if len(p_sents) >= 1 else [] for p_sents in sec_sents]

    # label of final sentence of topic is 1, final sentence of each paragraph is 0, other sentences is -100
    sec_sents = sum(sec_sents, [])
    sec_sent_labels = sum(sec_sent_labels, [])  # convert to 1-d list
    sec_sent_labels[-1] = 1

    return sec_sents, sec_sent_labels


def process_wiki_section_subset(train_file, dev_file, test_file, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    modes = ["train", "dev", "test"]
    files = [train_file, dev_file, test_file]
    all_mode_examples = {k: [] for k in modes}  # mode -> examples

    total_doc_cnt, total_topic_cnt, total_sent_cnt = 0, 0, 0
    for file_, mode in zip(files, modes):
        sent_cnt = 0
        topic_cnt = 0
        out_file = os.path.join(out_folder, mode + ".jsonl")
        res_examples = []
        with open(file_, "r") as f:
            data = json.load(f)
            for example_id, example in enumerate(data):
                text, annotations = example["text"], example["annotations"]
                sentences, labels = [], []
                section_topic_labels, sentence_topic_labels = [], []
                try:
                    for anno in annotations:
                        topic_cnt += 1
                        begin = anno["begin"]
                        length = anno["length"]

                        sec_text = text[begin:begin + length]
                        sec_sents, sec_sent_labels = tokenize_method(sec_text)
                        sentences += sec_sents
                        labels += sec_sent_labels
                        sent_cnt += len(sec_sents)

                        section_topic_labels.append(anno["sectionLabel"])
                        sentence_topic_labels += [anno["sectionLabel"]] * len(sec_sents)
                    if len(sentences) != len(labels):
                        print("example_id: {}, len(sentences): {} != len(labels): {}".format(example_id, len(sentences),
                                                                                             len(labels)))
                    assert len(sentences) == len(labels)
                except AssertionError:
                    print("ERROR: ", example_id)
                    continue
                json_example = {
                    "sentences": sentences,
                    "labels": labels,
                    "section_topic_labels": section_topic_labels,
                    "sentence_topic_labels": sentence_topic_labels,
                }
                res_examples.append(json.dumps(json_example) + "\n")

        with open(out_file, "w") as f:
            f.writelines(res_examples)
        doc_cnt = len(res_examples)

        all_mode_examples[mode] = res_examples
        total_sent_cnt += sent_cnt
        total_topic_cnt += topic_cnt
        total_doc_cnt += doc_cnt
        cnt_info = "mode: {}, doc_cnt: {}, topic_cnt: {}, sent_cnt: {}".format(mode, doc_cnt, topic_cnt, sent_cnt)
        print(cnt_info)
    print("total_doc_cnt: {}, total_topic_cnt: {}, total_sent_cnt: {}\n".format(total_doc_cnt, total_topic_cnt, total_sent_cnt))

    return all_mode_examples


def merge_wiki_section(out_folder, disease_mode_examples, city_mode_examples):
    wiki_section_train_file = os.path.join(out_folder, "train.jsonl")
    wiki_section_dev_file = os.path.join(out_folder, "dev.jsonl")
    wiki_section_test_file = os.path.join(out_folder, "test.jsonl")

    with open(wiki_section_train_file, "w") as f:
        f.writelines(disease_mode_examples["train"] + city_mode_examples["train"])
    with open(wiki_section_dev_file, "w") as f:
        f.writelines(disease_mode_examples["dev"] + city_mode_examples["dev"])
    with open(wiki_section_test_file, "w") as f:
        f.writelines(disease_mode_examples["test"] + city_mode_examples["test"])


def process_wiki_section(data_folder, out_folder):
    # process wiki_section en_disease
    disease_train_file = os.path.join(data_folder, "wikisection_en_disease_train.json")
    disease_dev_file = os.path.join(data_folder, "wikisection_en_disease_validation.json")
    disease_test_file = os.path.join(data_folder, "wikisection_en_disease_test.json")
    disease_out_folder = os.path.join(out_folder, "../wiki_section_disease")
    disease_mode_examples = process_wiki_section_subset(disease_train_file, disease_dev_file, disease_test_file, disease_out_folder)

    # process wiki_section en_city
    city_train_file = os.path.join(data_folder, "wikisection_en_city_train.json")
    city_dev_file = os.path.join(data_folder, "wikisection_en_city_validation.json")
    city_test_file = os.path.join(data_folder, "wikisection_en_city_test.json")
    city_out_folder = os.path.join(out_folder, "../wiki_section_city")
    city_mode_examples = process_wiki_section_subset(city_train_file, city_dev_file, city_test_file, city_out_folder)

    # merge en_disease and en_city to wiki_section
    merge_wiki_section(out_folder, disease_mode_examples, city_mode_examples)


def process_wiki_folder(folder, out_file):
    # get all files
    all_files = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            all_files.append(os.path.join(root, name))

    examples = []
    for file_ in tqdm(all_files):
        # get sections sentences and labels
        sentences, labels = [], []
        with open(file_, "r") as f:
            lines = f.readlines()
            sec_flag_indices = []
            for i, line in enumerate(lines):
                if line.startswith(sec_flag):
                    sec_flag_indices.append(i)
            sec_flag_indices.append(len(lines))

            for i in range(len(sec_flag_indices) - 1):
                start = sec_flag_indices[i] + 1
                end = sec_flag_indices[i + 1]
                if start == end:
                    # empty section
                    continue
                sec_sents = [line.strip() for line in lines[start:end]]
                sec_labels = [0] * len(sec_sents)
                sec_labels[-1] = 1

                sentences += sec_sents
                labels += sec_labels
        example = {
            "file": file_,
            "sentences": sentences,
            "labels": labels,
        }
        examples.append(json.dumps(example) + "\n")
    print("len(examples): ", len(examples))
    with open(out_file, "w") as f:
        f.writelines(examples)


def process_wiki727k(data_folder, out_folder):
    for mode in ["test", "dev", "train"]:
        folder = os.path.join(data_folder, mode)
        out_file = os.path.join(out_folder, mode + ".jsonl")
        print("out_file: ", out_file)
        process_wiki_folder(folder, out_file)


def process_wiki50(data_folder, out_folder):
    out_file = os.path.join(out_folder, "test.jsonl")
    process_wiki_folder(data_folder, out_file)


def process_wiki_elements(data_folder, out_folder):
    text_file = os.path.join(data_folder, "wikielements.text")
    seg_file = os.path.join(data_folder, "wikielements.segmenttitles")
    out_file = os.path.join(out_folder, "test.jsonl")

    with open(seg_file, "r") as f:
        seg_lines = f.readlines()
    with open(text_file, "r") as f:
        para_lines = f.readlines()
    assert len(seg_lines) == len(para_lines)

    doc_dict = {}
    for i, (seg_line, para_line) in enumerate(zip(seg_lines, para_lines)):
        doc_index, para_index, topic_title = seg_line.strip().split(",")[:3]  # index starts from 1
        if doc_index not in doc_dict:
            doc_dict[doc_index] = {"para_info": [], "topic_info": []}
        doc_dict[doc_index]["para_info"].append({
            "para_index": para_index,
            "topic_title": topic_title,
            "para_text": para_line.strip(),
        })

    doc_indices = sorted(doc_dict.keys())
    for doc_index in doc_indices:
        para_info = doc_dict[doc_index]["para_info"]
        seq_topic_labels = []
        cur_topic_title = ""
        for i in range(len(para_info) - 1, -1, -1):
            if para_info[i]["topic_title"] != cur_topic_title:
                seq_topic_labels.insert(0, 1)
            else:
                seq_topic_labels.insert(0, 0)
            cur_topic_title = para_info[i]["topic_title"]
        doc_dict[doc_index]["topic_info"] = {
            "sentences": [v["para_text"] for v in para_info],
            "labels": seq_topic_labels,
        }

    with open(out_file, "w") as f:
        for doc_index in doc_indices:
            f.write(json.dumps(doc_dict[doc_index]["topic_info"]) + "\n")


def get_data_name2folder(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    mapping = config['mapping']
    return mapping


def get_process_dict():
    process_dict = {
        "wiki_section": process_wiki_section,
        "wiki50": process_wiki50,
        "wiki727k": process_wiki727k,
        "wiki_elements": process_wiki_elements,
    }
    return process_dict


if __name__ == "__main__":
    out_root_folder = "../data"
    config_path = "./config/config.ini"
    data_name2folder = get_data_name2folder(config_path)
    process_dict = get_process_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", help="dataset to be processed")
    args = parser.parse_args()
    data_name = args.data_name

    data_folder = data_name2folder[data_name]
    out_folder = os.path.join(out_root_folder, data_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    process_fct = process_dict[data_name]
    process_fct(data_folder, out_folder)
