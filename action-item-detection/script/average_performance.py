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
"""Average performance from repeated experiments."""

import codecs
import os
import re
import sys
import numpy as np
from tqdm import tqdm


def get_file_metrics(file_path, metric_type="positive"):
    f1, precision, recall, accuracy = 0.0, 0.0, 0.0, 0.0
    checkpoint = 0
    threshold = 0.5
    result = []

    metric_start = "F1"
    metric_pattern = r"F1: ([\d\.]+)\%, Precision: ([\d\.]+)\%, Recall: ([\d\.]+)\%, Accuracy: ([\d\.]+)\%"
    if metric_type == "macro":
        metric_start = "Macro-F1"
        metric_pattern = r"Macro-F1: ([\d\.]+)\%, Macro-Precision: ([\d\.]+)\%, " \
                         r"Macro-Recall: ([\d\.]+)\%, Macro-Accuracy: ([\d\.]+)\%"
    elif metric_type == "positive":
        metric_start = "F1"
        metric_pattern = r"F1: ([\d\.]+)\%, Precision: ([\d\.]+)\%, Recall: ([\d\.]+)\%, Accuracy: ([\d\.]+)\%"

    check_start = "### Checkpoint"
    check_pattern = r"### Checkpoint: model.ckpt-(\d+), Threshold: ([\d\.]+)\%"

    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line.startswith(metric_start):
                line = line.strip()
                result.append(line)
                f1, precision, recall, accuracy = re.findall(metric_pattern, line)[0]
                f1 = float(f1) / 100.0
                precision = float(precision) / 100.0
                recall = float(recall) / 100.0
                accuracy = float(accuracy) / 100.0

            if line.startswith(check_start):
                checkpoint, threshold = re.findall(check_pattern, line)[0]
                checkpoint = int(checkpoint)
                threshold = float(threshold) / 100.0

            if line.startswith("Confusion"):
                line = line.strip()
                result.append(line)
    result = "; ".join(result)
    return f1, precision, recall, accuracy, result, checkpoint, threshold


def get_dir_metrics(file_path):
    f1, precision, recall, accuracy = 0.0, 0.0, 0.0, 0.0
    checkpoint = 0
    threshold = 0.5
    result = []

    metric_start = "Micro-F1"
    metric_pattern = r"Micro-F1: ([\d\.]+)\%, Micro-Precision: ([\d\.]+)\%, " \
                     r"Micro-Recall: ([\d\.]+)\%, Micro-Accuracy: ([\d\.]+)\%"

    check_start = "### Checkpoint"
    check_pattern = r"### Checkpoint: model.ckpt-(\d+), Threshold: ([\d\.]+)\%"

    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            if line.startswith(check_start):
                checkpoint, threshold = re.findall(check_pattern, line)[0]
                checkpoint = int(checkpoint)
                threshold = float(threshold) / 100.0
            if line.startswith(metric_start):
                line = line.strip()
                result.append(line)
                f1, precision, recall, accuracy = re.findall(metric_pattern, line)[0]
                f1 = float(f1) / 100.0
                precision = float(precision) / 100.0
                recall = float(recall) / 100.0
                accuracy = float(accuracy) / 100.0
            if line.startswith("All Confusion"):
                line = line.strip()
                result.append(line)
            if line.startswith("Macro-F1"):
                line = line.strip()
                result.append(line)
    result = "; ".join(result)
    return f1, precision, recall, accuracy, result, checkpoint, threshold


def average_performance(experiment_dirs, performance_file="performance.tsv", model_name="", file_name_list=None,
                        dir_name_list=None, return_type="best"):
    def get_performance(predict_type="file", predict_file="test", file_metric_type="positive", return_type="best"):
        assert return_type in ["best", "all", "set1", "set2", "set3", "set4"]
        f1_list, precision_list, recall_list, accuracy_list, result_list = [], [], [], [], []
        checkpoint_list, threshold_list = [], []
        experiment2metrics = {}
        for experiment_dir in experiment_dirs:
            predict_path = os.path.join(experiment_dir, f"predict/{predict_type}_{predict_file}/")
            predict_metric_file = os.path.join(predict_path, "test_metrics.txt")

            if not os.path.exists(predict_metric_file):
                continue

            f1, precision, recall, accuracy, result = 0.0, 0.0, 0.0, 0.0, ""
            checkpoint = 0
            threshold = 0.5
            if predict_type == "file":
                f1, precision, recall, accuracy, result, checkpoint, threshold = get_file_metrics(predict_metric_file, metric_type=file_metric_type)
            elif predict_type == "dir":
                f1, precision, recall, accuracy, result, checkpoint, threshold = get_dir_metrics(predict_metric_file)

            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            result_list.append(result)
            checkpoint_list.append(checkpoint)
            threshold_list.append(threshold)
            metric_item = {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy, "result": result,
                           "checkpoint": checkpoint, "threshold": threshold, "experiment": experiment_dir}
            experiment2metrics[experiment_dir] = metric_item

        if return_type == "all":
            return f1_list, precision_list, recall_list, accuracy_list, result_list, checkpoint_list, threshold_list

        fine_experiment2metrics = {}
        if return_type == "best":
            for experiment_dir, metric_item in experiment2metrics.items():
                fine_experiment_dir = experiment_dir.strip().split("_")[0]
                if fine_experiment_dir not in fine_experiment2metrics:
                    fine_experiment2metrics[fine_experiment_dir] = metric_item
                    continue
                fine_metric_item = fine_experiment2metrics[fine_experiment_dir]
                if metric_item["f1"] > fine_metric_item["f1"]:
                    fine_experiment2metrics[fine_experiment_dir] = metric_item
        else:  # set type
            for experiment_dir, metric_item in experiment2metrics.items():
                fine_experiment_dir = experiment_dir.strip().split("_")[0]
                if fine_experiment_dir in fine_experiment2metrics:
                    continue
                fine_key = f"{fine_experiment_dir}_{return_type}"
                fine_experiment2metrics[fine_experiment_dir] = experiment2metrics[fine_key]

        fine_f1_list, fine_precision_list, fine_recall_list, fine_accuracy_list = [], [], [], []
        fine_result_list, fine_checkpoint_list, fine_threshold_list = [], [], []
        for fine_experiment_dir, fine_metric_item in fine_experiment2metrics.items():
            fine_f1_list.append(fine_metric_item["f1"])
            fine_precision_list.append(fine_metric_item["precision"])
            fine_recall_list.append(fine_metric_item["recall"])
            fine_accuracy_list.append(fine_metric_item["accuracy"])
            fine_result_list.append(f"{fine_metric_item['experiment']}#{fine_metric_item['result']}")
            fine_checkpoint_list.append(fine_metric_item["checkpoint"])
            fine_threshold_list.append(fine_metric_item["threshold"])

        return fine_f1_list, fine_precision_list, fine_recall_list, fine_accuracy_list, fine_result_list, fine_checkpoint_list, fine_threshold_list

    def convert2array(f1_list, precision_list, recall_list, accuracy_list):
        f1_list = np.asarray(f1_list)
        precision_list = np.asarray(precision_list)
        recall_list = np.asarray(recall_list)
        accuracy_list = np.asarray(accuracy_list)
        return f1_list, precision_list, recall_list, accuracy_list

    def get_average(f1_list, precision_list, recall_list, accuracy_list):
        avg_f1 = np.mean(f1_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_accuracy = np.mean(accuracy_list)
        return avg_f1, avg_precision, avg_recall, avg_accuracy

    def get_variance(f1_list, precision_list, recall_list, accuracy_list):
        var_f1 = np.var(f1_list)
        var_precision = np.var(precision_list)
        var_recall = np.var(recall_list)
        var_accuracy = np.var(accuracy_list)
        return var_f1, var_precision, var_recall, var_accuracy

    def get_standard_deviation(f1_list, precision_list, recall_list, accuracy_list):
        std_f1, std_precision, std_recall, std_accuracy = 0.0, 0.0, 0.0, 0.0
        if len(f1_list) == 1:
            return std_f1, std_precision, std_recall, std_accuracy
        std_f1 = np.std(f1_list, ddof=1)
        std_precision = np.std(precision_list, ddof=1)
        std_recall = np.std(recall_list, ddof=1)
        std_accuracy = np.std(accuracy_list, ddof=1)
        return std_f1, std_precision, std_recall, std_accuracy

    def get_maximum(f1_list, precision_list, recall_list, accuracy_list):
        max_f1 = np.max(f1_list)
        max_precision = np.max(precision_list)
        max_recall = np.max(recall_list)
        max_accuracy = np.max(accuracy_list)
        return max_f1, max_precision, max_recall, max_accuracy

    def get_best(f1_list, precision_list, recall_list, accuracy_list):
        best_f1, best_precision, best_recall, best_accuracy = -1, -1, -1, -1
        for f1, precision, recall, accuracy in zip(f1_list, precision_list, recall_list, accuracy_list):
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_accuracy = accuracy
        return best_f1, best_precision, best_recall, best_accuracy

    def get_minimum(f1_list, precision_list, recall_list, accuracy_list):
        min_f1 = np.min(f1_list)
        min_precision = np.min(precision_list)
        min_recall = np.min(recall_list)
        min_accuracy = np.min(accuracy_list)
        return min_f1, min_precision, min_recall, min_accuracy

    def output_performance(fw, predict_type="file", predict_file="test", model_name="bert_test", file_metric_type="positive", return_type="best"):
        f1_list, precision_list, recall_list, accuracy_list, result_list, checkpoint_list, threshold_list = \
            get_performance(predict_type=predict_type, predict_file=predict_file, file_metric_type=file_metric_type,
                            return_type=return_type)
        if len(f1_list) <= 0:
            return

        f1_list, precision_list, recall_list, accuracy_list = convert2array(f1_list, precision_list, recall_list,
                                                                            accuracy_list)
        avg_f1, avg_precision, avg_recall, avg_accuracy = get_average(f1_list, precision_list, recall_list,
                                                                      accuracy_list)
        # var_f1, var_precision, var_recall, var_accuracy = get_variance(f1_list, precision_list, recall_list, accuracy_list)
        std_f1, std_precision, std_recall, std_accuracy = get_standard_deviation(f1_list, precision_list, recall_list, accuracy_list)
        max_f1, max_precision, max_recall, max_accuracy = get_maximum(f1_list, precision_list, recall_list, accuracy_list)
        min_f1, min_precision, min_recall, min_accuracy = get_minimum(f1_list, precision_list, recall_list, accuracy_list)
        best_f1, best_precision, best_recall, best_accuracy = get_best(f1_list, precision_list, recall_list, accuracy_list)

        result = "RESULT"
        model_name = model_name.replace("_", "-")
        test_name = f"{predict_type}#{predict_file}"
        if predict_type == "file":
            test_name += f"#{file_metric_type}"
        elif predict_type == "dir":
            test_name += f"#micro"

        avg_result = f"{avg_f1 * 100:.2f}/{avg_precision * 100:.2f}/{avg_recall * 100:.2f}"
        std_result = f"{std_f1 * 100:.2f}/{std_precision * 100:.2f}/{std_recall * 100:.2f}"
        best_result = f"{best_f1 * 100:.2f}/{best_precision * 100:.2f}/{best_recall * 100:.2f}"
        max_result = f"{max_f1 * 100:.2f}/{max_precision * 100:.2f}/{max_recall * 100:.2f}"
        min_result = f"{min_f1 * 100:.2f}/{min_precision * 100:.2f}/{min_recall * 100:.2f}"

        fw.write(f"{model_name}\t平均值\t{test_name}\t{avg_f1:.2%}\t{avg_precision:.2%}\t{avg_recall:.2%}\t{avg_accuracy:.2%}\t{avg_result}\n")
        fw.write(f"{model_name}\t标准差\t{test_name}\t{std_f1:.2%}\t{std_precision:.2%}\t{std_recall:.2%}\t{std_accuracy:.2%}\t{std_result}\n")
        fw.write(f"{model_name}\t最优值\t{test_name}\t{best_f1:.2%}\t{best_precision:.2%}\t{best_recall:.2%}\t{best_accuracy:.2%}\t{best_result}\n")
        fw.write(f"{model_name}\t最大值\t{test_name}\t{max_f1:.2%}\t{max_precision:.2%}\t{max_recall:.2%}\t{max_accuracy:.2%}\t{max_result}\n")
        fw.write(f"{model_name}\t最小值\t{test_name}\t{min_f1:.2%}\t{min_precision:.2%}\t{min_recall:.2%}\t{min_accuracy:.2%}\t{min_result}\n")

        fw.write("\n")
        for idx, (f1, precision, recall, accuracy, result, checkpoint, threshold) in \
                enumerate(zip(f1_list, precision_list, recall_list, accuracy_list, result_list, checkpoint_list, threshold_list)):
            model_type = f"模型{idx + 1}#{checkpoint}#{threshold}"
            fw.write(f"{model_name}\t{model_type}\t{test_name}\t{f1:.2%}\t{precision:.2%}\t{recall:.2%}\t{accuracy:.2%}\t{result}\n")
        fw.write("\n\n")

    with codecs.open(performance_file, "w", encoding="utf-8") as fw:
        fw.write("模型名称\t类型\t测试集名称\tF1\tP\tR\tA\t模型表现\n\n")
        file_metric_types = ["positive", "macro"]

        for file_name in file_name_list:
            for file_metric_type in file_metric_types:
                output_performance(fw, predict_type="file", predict_file=file_name, model_name=model_name, file_metric_type=file_metric_type, return_type=return_type)

        for dir_name in dir_name_list:
            output_performance(fw, predict_type="dir", predict_file=dir_name, model_name=model_name)


if __name__ == "__main__":
    num_experiment = 1
    num_experiment_set = 4
    return_fine_type = "best"  # best, all, set1, set2, ...

    if len(sys.argv) == 2:
        num_experiment = int(sys.argv[1])
    if len(sys.argv) == 3:
        num_experiment = int(sys.argv[1])
        num_experiment_set = int(sys.argv[2])
    if len(sys.argv) == 4:
        num_experiment = int(sys.argv[1])
        num_experiment_set = int(sys.argv[2])
        return_fine_type = sys.argv[3]

    experiment_dir_list = []
    if num_experiment == 1:
        experiment_dir_list = ["best_model"]
    else:
        for i in range(1, num_experiment + 1):
            for j in range(1, num_experiment_set + 1):
                experiment_dir_list.append(f"best{i}_set{j}")

    model_name = "bert_test"

    performance_file = f"experiment_{return_fine_type}.tsv"

    file_name_list = ["test", "dev"]
    dir_name_list = []

    average_performance(experiment_dir_list, performance_file, model_name, file_name_list, dir_name_list,
                        return_type=return_fine_type)











