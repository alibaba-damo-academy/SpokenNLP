
import os
import argparse
import json
from datasets import load_metric


def read_total_pred_and_labels(data_file, pred_file):
    total_para_level_predictions = []
    total_para_level_labels = []
    with open(pred_file, "r") as f:
        for line in f.readlines():
            tmp = json.loads(line.strip())
            para_level_labels = [0 if v == "O" else 1 for v in tmp["labels"]]   # 1 means seg
            total_para_level_labels.append(para_level_labels)
            para_level_predictions = [0 if v == "O" else 1 for v in tmp["predictions"]]
            total_para_level_predictions.append(para_level_predictions)
    
    total_sent_level_labels = []    # without last boundary of each example
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = json.loads(line.strip())
            sent_level_labels = tmp["labels"]
            total_sent_level_labels.append(sent_level_labels[:-1])
    
    return total_para_level_predictions, total_para_level_labels, total_sent_level_labels
    

def get_wiki_section_sent_level_metric(data_file, pred_file):
    disease_cnt = 718
    city_cnt = 3893

    total_para_level_predictions, total_para_level_labels, total_sent_level_labels = read_total_pred_and_labels(data_file, pred_file)
    assert len(total_para_level_predictions) == disease_cnt + city_cnt
    
    a = [[v for v in x] for x in total_para_level_predictions]
    b = [[v for v in x] for x in total_para_level_labels]
    c = [[v for v in x] for x in total_sent_level_labels]    

    disease_para_level_predictions, disease_para_level_labels, disease_sent_level_labels = \
    total_para_level_predictions[:disease_cnt], total_para_level_labels[:disease_cnt], total_sent_level_labels[:disease_cnt]

    city_para_level_predictions, city_para_level_labels, city_sent_level_labels = \
    total_para_level_predictions[disease_cnt:], total_para_level_labels[disease_cnt:], total_sent_level_labels[disease_cnt:]

    print(" / ".join(["p", "r", "f1", "pk", "wd"]))
    get_sent_level_metric_from_para_level_models("wiki_section_disease", disease_para_level_predictions, disease_para_level_labels, disease_sent_level_labels)
    get_sent_level_metric_from_para_level_models("wiki_section_city", city_para_level_predictions, city_para_level_labels, city_sent_level_labels)
    get_sent_level_metric_from_para_level_models("wiki_section", a, b, c)


def get_sent_level_metric_from_para_level_models(data_name, total_para_level_predictions, total_para_level_labels, total_sent_level_labels):
    total_sent_level_predictions = []
    for example_index, (para_level_labels, sent_level_labels, para_level_predictions) in enumerate(zip(total_para_level_labels, total_sent_level_labels, total_para_level_predictions)):
        assert len(para_level_labels) == len([v for v in sent_level_labels if v != -100])

        sent_level_predictions = [0] * len(sent_level_labels)
        p_id = 0
        for i in range(len(sent_level_labels)):
            if sent_level_labels[i] != -100:        # paragraph point
                assert sent_level_labels[i] == para_level_labels[p_id]
                sent_level_predictions[i] = para_level_predictions[p_id]
                p_id += 1
            else:
                sent_level_labels[i] = 0
        total_sent_level_labels[example_index] = sent_level_labels
        total_sent_level_predictions.append(sent_level_predictions)
    
    evaluator = load_metric("./src/metrics/seqeval.py")
    print("data_name: ", data_name)
    res = evaluator.compute_window_metric(total_sent_level_predictions, total_sent_level_labels)
    print("sent_level: " + " / ".join(["%.2f" % (v * 100) for v in [res["precision"], res["recall"], res["f1"], res["pk"], res["wd"]]]))
    para_res = evaluator.compute_window_metric(total_para_level_predictions, total_para_level_labels)
    print("para_level: " + " / ".join(["%.2f" % (v * 100) for v in [para_res["precision"], para_res["recall"], para_res["f1"], para_res["pk"], para_res["wd"]]]))
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="data_file")
    parser.add_argument("pred_file", help="pred_file")
    args = parser.parse_args()
    data_file = args.data_file
    pred_file = args.pred_file

    print("data_file: {}".format(data_file))
    print("pred_file: {}".format(pred_file))
    print("get_wiki_section_sent_level_metric")
    get_wiki_section_sent_level_metric(data_file, pred_file)
