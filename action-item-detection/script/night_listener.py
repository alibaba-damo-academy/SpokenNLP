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
"""Saving best model during evaluation."""

import os
import re
import sys
import time
import logging


def check_path(path):
    if not os.path.exists(path):
        cmd = "mkdir {}".format(path)
        os.system(cmd)


def get_new_info(origin_log_file, best_f1):
    global_step = -1
    metrics = {}
    if not os.path.exists(origin_log_file):
        return -1, best_f1, metrics
    log_prefix = ".".join(os.path.basename(origin_log_file).split(".")[:-1])
    log_file = os.path.join(os.path.dirname(origin_log_file), "{}.copy.txt".format(log_prefix))
    cp_file_cmd = "cp {} {}".format(origin_log_file, log_file)
    os.system(cp_file_cmd)
    with open(log_file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            eval_pattern = r"Saving dict for global step (\d+): eval/accuracy = (\d\.\d+), eval/f1 = (\d\.\d+), eval/loss = (\d+\.\d+), eval/precision = (\d\.\d+), eval/recall = (\d\.\d+), global_step"
            result = re.findall(eval_pattern, line)
            if result:
                global_step = int(result[0][0])
                accuracy, f1, loss, precision, recall = [float(metric) for metric in result[0][1:]]
    if global_step > 0 and f1 > best_f1:
        best_f1 = f1
        metrics["global_step"] = global_step
        metrics["loss"] = loss
        metrics["accuracy"] = accuracy
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        output_dir = os.path.basename(os.path.dirname(origin_log_file).strip("/"))
        logging.info(f"## Get best F1 score for {output_dir} at {global_step} step")
        logging.info("## F1 Score: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, Accuracy: {:.2%}, Loss: {:.4f}\n".
                     format(f1, precision, recall, accuracy, loss))
        return global_step, best_f1, metrics
    return -1, best_f1, metrics


def backup_best_model(best_model_dir, output_dir, best_ckpt, back_up_num=5):

    def get_ckpts(model_dir):
        files = [f for f in os.listdir(model_dir) if f.startswith("model.ckpt")]
        ckpts = list(set([int(f.split("-")[1].split(".")[0]) for f in files]))
        return sorted(ckpts)

    output_ckpts = get_ckpts(output_dir)
    if best_ckpt in output_ckpts:
        cp_model_cmd = "cp {}/model.ckpt-{}.* {}/".format(output_dir, best_ckpt, best_model_dir)
        os.system(cp_model_cmd)
        logging.info(cp_model_cmd)
        cp_checkpoint_cmd = "cp {}/checkpoint {}/".format(output_dir, best_model_dir)
        os.system(cp_checkpoint_cmd)
        logging.info(cp_checkpoint_cmd)

    best_ckpts = get_ckpts(best_model_dir)
    while len(best_ckpts) > back_up_num:
        old_ckpt = best_ckpts[0]
        rm_model_cmd = "rm {}/model.ckpt-{}.*".format(best_model_dir, old_ckpt)
        os.system(rm_model_cmd)
        logging.info(rm_model_cmd)
        best_ckpts = get_ckpts(best_model_dir)


def save_metrics(best_model_dir, best_ckpt, metrics):
    metric_file = os.path.join(best_model_dir, "model.ckpt-{}.metric.txt".format(best_ckpt))
    with open(metric_file, "w", encoding="utf-8") as fw:
        for metric, value in metrics.items():
            fw.write("{} = {}\n".format(metric, value))


def night_listen(output2best_map, best2f1_map, back_up_num=3):
    while True:
        time.sleep(2)
        for output_dir, best_model_dir in output2best_map.items():
            output_log_file = os.path.join(output_dir, "model.log.txt")
            best_f1 = best2f1_map[best_model_dir]
            best_ckpt, best_f1, metrics = get_new_info(output_log_file, best_f1)
            best2f1_map[best_model_dir] = best_f1
            if best_ckpt > 0:
                backup_best_model(best_model_dir, output_dir, best_ckpt, back_up_num)
                save_metrics(best_model_dir, best_ckpt, metrics)


if __name__ == '__main__':
    repeat_time = 1
    num_set = 4
    num_back_up = 3
    if len(sys.argv) == 2:
        repeat_time = int(sys.argv[1])
    elif len(sys.argv) == 3:
        repeat_time = int(sys.argv[1])
        num_set = int(sys.argv[2])
    elif len(sys.argv) == 4:
        repeat_time = int(sys.argv[1])
        num_set = int(sys.argv[2])
        num_back_up = int(sys.argv[3])
    elif len(sys.argv) > 4:
        print("Please: python night_listener.py <repeat_time> <num_set> <num_back_up>")

    output2best_dict = {}  # key: output_dir, value: best_model_dir
    best2f1_dict = {}  # key: best_model_dir, value: f1

    if repeat_time == 1:
        output_dir = f"./output/"
        best_model_dir = f"./best_model/"
        check_path(best_model_dir)
        output2best_dict[output_dir] = best_model_dir
        best2f1_dict[best_model_dir] = -1.0
    else:
        for idx in range(1, repeat_time + 1):
            for set_idx in range(1, num_set + 1):
                output_dir = f"./output{idx}_set{set_idx}/"
                best_model_dir = f"./best{idx}_set{set_idx}/"
                check_path(best_model_dir)
                output2best_dict[output_dir] = best_model_dir
                best2f1_dict[best_model_dir] = -1.0

    logging_file = "best_model.log.txt"
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(logging_file, mode="a"),
                                  logging.StreamHandler()
                                  ])

    night_listen(output2best_dict, best2f1_dict, back_up_num=num_back_up)





