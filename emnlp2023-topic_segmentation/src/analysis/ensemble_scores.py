
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import math
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc
from datasets import ClassLabel, load_dataset, load_metric


def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


def read_predictions(pred_file, return_cos_sim=False, sim_temp=1):
    total_labels, total_probs, total_cos_sims = [], [], []
    total_predic_logits = []
    example_seq_lengths = []
    with open(pred_file, "r") as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            res = json.loads(line)
            labels = [0 if v == "O" else 1 for v in res["labels"]]      # paragraph level. 1 means seg
            predict_logits = res["predict_logits"]
            eop_pair_cos_sim = [v * sim_temp for v in res["eop_pair_cos_sim"]]
            total_predic_logits += predict_logits
            total_labels += labels
            total_cos_sims += eop_pair_cos_sim
            example_seq_lengths.append(len(labels))
    total_probs = scipy.special.softmax(np.array(total_predic_logits), axis=-1)[:, 0].tolist()     # [:, 0] means get position 0 score, which is seg prob
      
    if return_cos_sim:
        return total_labels, total_probs, total_cos_sims, example_seq_lengths
    else:
        return total_labels, total_probs, example_seq_lengths


def ensemble_scores(pred_file, sim_temp=1):
    test_labels, test_probs, test_cos_sims, test_example_seq_lengths = read_predictions(pred_file, return_cos_sim=True, sim_temp=sim_temp)
    print("number of test_labels: ", len(test_labels), " number of 1 in test_labels: ", sum(test_labels))
    
    print("\nprob only (threshold=0.5): ")
    total_pred_by_prob_only = [1 if v > 0.5 else 0 for v in test_probs]        # 1 means seg
    print("f1: %.2f, p: %.2f, r: %.2f" % (100 * f1_score(test_labels, total_pred_by_prob_only), 100 * precision_score(test_labels, total_pred_by_prob_only), 100 * recall_score(test_labels, total_pred_by_prob_only)))
    
    print("\ncos_sim only (threshold=0.0): ")
    total_pred_by_cos_sim_only = [1 if v <= 0.0 else 0 for v in test_cos_sims]
    print("f1: %.2f, p: %.2f, r: %.2f" % (100 * f1_score(test_labels, total_pred_by_cos_sim_only), 100 * precision_score(test_labels, total_pred_by_cos_sim_only), 100 * recall_score(test_labels, total_pred_by_cos_sim_only)))

    print("\n(prob + sigmoid(-1 * cos_sim)) / 2 (threshold = %.2f): " % 0.5)
    total_preds = [1 if (a + stable_sigmoid(-1 * b)) / 2 > 0.5 else 0 for a, b in zip(test_probs, test_cos_sims)]        # 1 means seg
    print("f1: %.2f, p: %.2f, r: %.2f" % (100 * f1_score(test_labels, total_preds), 100 * precision_score(test_labels, total_preds), 100 * recall_score(test_labels, total_preds)))
    
    all_examples_predictions, all_example_labels = [], []
    assert len(test_labels) == sum(test_example_seq_lengths)     # var test_labels is test_labels
    cur_index = 0
    for v in test_example_seq_lengths:
        all_examples_predictions.append(total_preds[cur_index : cur_index + v])
        all_example_labels.append(test_labels[cur_index : cur_index + v])
        cur_index += v
    
    evaluator = load_metric("../metrics/seqeval.py")
    res = evaluator.compute_window_metric(all_examples_predictions, all_example_labels)
    print(res)
    print(" / ".join(["%.2f" % (v * 100) for v in [res["precision"], res["recall"], res["f1"], res["pk"], res["wd"]]]))
 

 if __name__ == "__main__":
    pred_file = "path/to/the/pred_file"
    ensemble_scores(pred_file, sim_temp=1)