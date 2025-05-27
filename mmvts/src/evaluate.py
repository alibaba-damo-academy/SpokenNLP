
import os
import argparse
import json
import numpy as np
import statistics

from scipy.special import softmax
from sklearn.metrics import f1_score, precision_score, recall_score


types = ["ppt", "blackboard", "other"]
en_type2video = {
    'blackboard': ['mit001', 'mit002', 'mit035', 'mit038', 'mit126', 'mit159', 'mit151', 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 56, 57, 58, 59, 60, 61, 62], 
    'ppt': ['mit088', 'mit097', 'mit153', 'mit049', 'mit039', 'mit032', 0, 1, 2, 10, 11, 12, 13, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 49, 50, 51, 52, 53, 54, 55], 
    'other': ['mit057', 'mit075', 27, 28, 63, 64, 65, 66, 67, 68, 69]
}


def get_cn_type2video(cn_type2video_file):
    with open(cn_type2video_file, "r") as f:
        res = json.load(f)
    return res 
cn_type2video_file = "/home/yuhai.yu/lvts/data/visual_related/cn_course_0523/cn_course_0523_type2videos.json"
cn_type2video = get_cn_type2video(cn_type2video_file)
cn_data_file = "/home/yuhai.yu/lvts/data/visual_related/clvts/test.jsonl"
en_data_file = "/home/yuhai.yu/lvts/data/avlecture/avlecture/test.jsonl"

def read_jsonl_file(in_file):
    data = []
    with open(in_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def get_topk_preds_purely(labels, probs, topk):
    if topk == -1:
        sel_topk = sum(labels)
    else:
        sel_topk = topk
    topk_indices = np.argsort(probs)[-sel_topk:]
    topk_preds = np.zeros_like(labels)
    topk_preds[topk_indices] = 1
    topk_preds = list(topk_preds)
    return topk_preds


def get_topk_preds_like_texttiling(labels, probs, stet, topk=-1, time_span=30):
    # current labels and probs don't contain last boundary
    # get indices of topK probs like texttiling, specifically when loop probs from high to low, 
    # if there exists clip within 30 seconds has been set to 1, then current position is 0 and is not included in topK.
    preds = np.array([[a, b[1]] for a, b in zip(probs, stet)])  # prob, clip_end_second
    if topk == -1:
        topk = sum(labels)

    topk_indices = np.argsort(preds[:, 0])[::-1]  # eg. probs = [(3,15),(4,21), (2,2)] topk_indices=[1, 0, 2]
    topk_preds = np.zeros_like(labels)

    choose_cnt = 0
    for i in topk_indices:
        if choose_cnt == topk:
            break
        cur_prob, cur_end_second = preds[i]
        if cur_prob != -1:
            # It means that there is no higher probability value nearby.
            topk_preds[i] = 1
            choose_cnt += 1

            # set the probability value near it to -1
            right = i + 1
            while right < len(preds) and preds[right][1] - cur_end_second <= time_span:
                preds[right][0] = -1
                right += 1
            left = i - 1
            while left >= 0 and cur_end_second - preds[left][1] <= time_span:
                preds[left][0] = -1
                left -= 1

    topk_preds_like_texttiling = list(topk_preds)
    return topk_preds_like_texttiling


def get_llm_result(data_file, pred_file):
    res = []
    prediction_key = "predict"
    all_data = read_jsonl_file(data_file)
    all_pred = read_jsonl_file(pred_file)
    for example_info, pred_info in zip(all_data, all_pred):
        labels = example_info["labels"][:-1]        # remove last boundary temporarily
        predictions = pred_info[prediction_key][:len(labels)]

        label_seconds = example_info["topic_end_seconds"]
        threshold_seconds = [example_info["stet"][i][1] for i, l in enumerate(predictions) if l == 1]

        labels.append(1)
        predictions.append(1)
        threshold_seconds.append(label_seconds[-1])

        res.append({
            "example_id": example_info["example_id"] if "example_id" in example_info else "orig_example_id",
            "labels": labels,
            "label_seconds": [int(v) for v in label_seconds],
            "threshold_preds": predictions,
            "threshold_seconds": [int(v) for v in threshold_seconds],
        })

    return res


def get_pred_result(data_file, pred_file, topk=-1, logit_type="mm"):
    res = []

    if logit_type == "mm":
        prediction_key = "predictions"
        logit_key = "predict_logits"
    elif logit_type == "text":
        prediction_key = "text_predictions"
        logit_key = "text_logits"
    elif logit_type == "vis":
        prediction_key = "vis_predictions"
        logit_key = "vis_logits"
    else:
        raise ValueError("not supported logit_type")

    all_data = read_jsonl_file(data_file)
    all_pred = read_jsonl_file(pred_file)

    for example_info, pred_info in zip(all_data, all_pred):
        labels = example_info["labels"][:-1]        # remove last boundary temporarily
        predictions = [0 if p == "O" or p == 0 else 1 for p in pred_info[prediction_key]]
        logits = pred_info[logit_key]

        label_seconds = example_info["topic_end_seconds"]
        threshold_seconds = [example_info["stet"][i][1] for i, l in enumerate(predictions) if l == 1]
        # 如果某个clip被预测为1，那么该clip内的所有秒都会被计算到bs@30中，而不是只让clip的结束时间戳被计算到bs@30
        threshold_seconds_v2 = sum([list(range(int(example_info["stet"][i][0]), int(example_info["stet"][i][1]) + 1)) for i, l in enumerate(predictions) if l == 1], [])

        softmax_output = softmax(np.array(logits), axis=1)
        probs = softmax_output[:, 0]
        topk_preds = get_topk_preds_like_texttiling(labels, probs, example_info["stet"], topk)
        topk_seconds = [example_info["stet"][i][1] for i, l in enumerate(topk_preds) if l == 1]
        topk_seconds_v2 = sum([list(range(int(example_info["stet"][i][0]), int(example_info["stet"][i][1]) + 1)) for i, l in enumerate(topk_preds) if l == 1], [])

        # include last boundary
        labels.append(1)
        predictions.append(1)
        topk_preds.append(1)
        threshold_seconds.append(label_seconds[-1])
        topk_seconds.append(label_seconds[-1])
        threshold_seconds_v2.append(label_seconds[-1])
        topk_seconds_v2.append(label_seconds[-1])

        res.append({
            "example_id": example_info["example_id"] if "example_id" in example_info else "orig_example_id",
            "labels": labels,
            "label_seconds": [int(v) for v in label_seconds],
            "threshold_preds": predictions,
            "threshold_seconds": [int(v) for v in threshold_seconds],
            "topk_preds": topk_preds,
            "topk_seconds": [int(v) for v in topk_seconds],
            "pos_scores": probs,

            "threshold_seconds_v2": threshold_seconds_v2,
            "topk_seconds_v2": topk_seconds_v2,
        })

    return res


def get_bs_at_k(label_end_seconds, pred_end_seconds, threshold=30):
    def closest_number1(arr, k, th):
        for i, val in enumerate(arr):
            if abs(val - k) < th:
                return i
        return -1

    cnt = 0
    num_segments = len(label_end_seconds)
    pred_segments = len(pred_end_seconds)
    assert num_segments >= 1

    help_label_end_seconds = [v for v in label_end_seconds]
    for i_p in pred_end_seconds:
        idx = closest_number1(help_label_end_seconds, i_p, threshold)
        if idx == -1:
            continue
        help_label_end_seconds[idx] = -1000000000
        cnt += 1
    bs_score = cnt / num_segments
    return bs_score, cnt - 1, num_segments - 1      # 返回该样本的边界数和命中数（除去最后一个边界）


def for_f1_tolerance(label_end_seconds, pred_end_seconds, threshold=30):
    def closest_number1(arr, k, th):
        for i, val in enumerate(arr):
            if abs(val - k) < th:
                return i
        return -1

    hit_cnt = 0
    num_segments = len(label_end_seconds)
    pred_segments = len(pred_end_seconds)
    assert num_segments >= 1

    help_label_end_seconds = [v for v in label_end_seconds]
    for i_p in pred_end_seconds:
        idx = closest_number1(help_label_end_seconds, i_p, threshold)
        if idx == -1:
            continue
        help_label_end_seconds[idx] = -1000000000
        hit_cnt += 1
    return hit_cnt - 1, num_segments - 1, pred_segments - 1      # 除去最后一个边界，返回命中数，label数和预测数



def get_miou_by_overlap(label_end_seconds, pred_end_seconds):
    # https://github.com/kakaobrain/bassl/blob/main/bassl/finetune/utils/metric.py

    def end_seconds_list2stet_list(seconds_list):
        #  [10, 30, 34] -> [(0, 10), (10, 30), (30, 34)]
        tmp = [v for v in seconds_list]
        tmp.insert(0, 0)
        res = []
        for i in range(1, len(tmp)):
            res.append((tmp[i - 1], tmp[i]))
        return res

    def cal_miou(gt_list, pred_list):
        def getIntersection(interval_1, interval_2):
            # assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
            # assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
            start = max(interval_1[0], interval_2[0])
            end = min(interval_1[1], interval_2[1])
            if start < end:
                return end - start
            return 0

        def getUnion(interval_1, interval_2):
            assert interval_1[0] < interval_1[1], "start frame is bigger than end frame."
            assert interval_2[0] < interval_2[1], "start frame is bigger than end frame."
            start = min(interval_1[0], interval_2[0])
            end = max(interval_1[1], interval_2[1])
            return end - start

        def getRatio(interval_1, interval_2):
            interaction = getIntersection(interval_1, interval_2)
            if interaction == 0:
                return 0
            else:
                return interaction / getUnion(interval_1, interval_2)

        mious = []
        for gt_item in gt_list:
            rats = []
            for pred_item in pred_list:
                rat = getRatio(pred_item, gt_item)
                rats.append(rat)
            mious.append(np.max(rats))
        miou = np.mean(mious)
        return miou

    label_stets = end_seconds_list2stet_list(label_end_seconds)
    pred_stets = end_seconds_list2stet_list(pred_end_seconds)

    miou1 = cal_miou(label_stets, pred_stets)
    miou2 = cal_miou(pred_stets, label_stets)
    return np.mean([miou1, miou2])


def ecls_to_ts(ecls_cluster):
    # second level topic id sequence -> topic duration
    # [0, 0, 0, 1, 1, 2, 2, 2, 2] -> [3, 2, 4]
    ts = [0] * (max(ecls_cluster) + 1)
    for i in ecls_cluster:
        ts[i] += 1
    return np.array(ts)


def get_clip_f1(label_seq, pred_seq):
    return f1_score(label_seq[:-1], pred_seq[:-1])
    

def get_score(label_topic_ids, pred_topic_ids, label_seq, pred_seq, pred_seconds_v2, bs_threshold=30):
    # label_topic_ids, pred_topic_ids are second level topic id, eg. [0, 0, 0, 1, 1, 2, 2, 2, 2] means 9 seconds.
    # label_seq and pred_seq are clip level 0 1 label.
    label_topic_ids = np.array(label_topic_ids)
    pred_topic_ids = np.array(pred_topic_ids)

    label_end_seconds = list(np.cumsum(ecls_to_ts(label_topic_ids)))          # label end timestamp of each topic
    pred_end_seconds = list(np.cumsum(ecls_to_ts(pred_topic_ids)))            # pred end timestamp of each topic

    bs_score, bs_hit, bs_total = get_bs_at_k(label_end_seconds, pred_end_seconds, bs_threshold)
    miou_by_overlap = get_miou_by_overlap(label_end_seconds, pred_end_seconds)
    clip_f1 = get_clip_f1(label_seq, pred_seq)
    hit_num, actual_num, pred_num = for_f1_tolerance(label_end_seconds, pred_end_seconds, bs_threshold)

    bs_score_v2, bs_hit_v2, bs_total_v2, = get_bs_at_k(label_end_seconds, pred_seconds_v2, bs_threshold)
    return {
        "bs@{}".format(bs_threshold): bs_score,
        "miou": miou_by_overlap,
        "clip_f1": clip_f1,
        "bs@{}v2".format(bs_threshold): bs_score_v2,

        "bs_hit": bs_hit,
        "bs_total": bs_total,
        "bs_hit_v2": bs_hit_v2,
        "bs_total_v2": bs_total_v2,

        "f1_tolerance_hit": hit_num,
        "f1_tolerance_actual": actual_num,
        "f1_tolerance_pred": pred_num
    }


def seconds2clusters(seconds):
    # in: [4, 7, 12, 20]    ith value means end timestamp of ith topic
    # out: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    if len(seconds) == 1:
        return [0] * seconds[0]
    durations = [seconds[0]]
    for i in range(1, len(seconds)):
        durations.append(seconds[i] - seconds[i - 1])

    res = []
    for i, dur in enumerate(durations):
        res += [i] * dur
    return np.array(res)


def compute_scores_llm(pred_res, bs_metric_key="bs@30"):
    bs_threshold = int(bs_metric_key.split("@")[1])

    th_bs_list, th_miou_by_overlap_list, th_clip_f1_list, th_bs_v2_list = [], [], [], []
    whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq = [], [], []
    th_bs_hit, th_bs_total, th_bs_hit_v2, th_bs_total_v2 = 0, 0, 0, 0

    th_f1_tolerance_hit, th_f1_tolerance_actual, th_f1_tolerance_pred = 0, 0, 0
    for index, example in enumerate(pred_res):
        example_id = example["example_id"]

        labels = example["labels"]
        label_seconds = example["label_seconds"]

        threshold_preds = example["threshold_preds"]
        if len(labels) != len(threshold_preds):
            print("index: {}, labels: {}, preds: {}".format(index, len(labels), len(threshold_preds)))
            continue
        
        whole_label_seq.append(labels)
        whole_th_pred_seq.append(threshold_preds)
        threshold_seconds = example["threshold_seconds"]

        label_clusters = seconds2clusters(label_seconds)
        threshold_clusters = seconds2clusters(threshold_seconds)

        label_seconds_from_clusters = list(np.cumsum(ecls_to_ts(label_clusters)))
        # print("label_seconds: {}, label_seconds_from_clusters: {}".format(label_seconds, label_seconds_from_clusters))
        # assert label_seconds == label_seconds_from_clusters

        th_score = get_score(label_clusters, threshold_clusters, labels, threshold_preds, threshold_preds, bs_threshold=bs_threshold)
        th_bs, th_miou, th_clip_f1 = th_score[bs_metric_key], th_score["miou"], th_score["clip_f1"]
        th_bs_list.append(th_bs)
        th_miou_by_overlap_list.append(th_miou)
        
        th_bs_hit += th_score["bs_hit"]
        th_bs_total += th_score["bs_total"]

        th_f1_tolerance_hit += th_score["f1_tolerance_hit"]
        th_f1_tolerance_actual += th_score["f1_tolerance_actual"]
        th_f1_tolerance_pred += th_score["f1_tolerance_pred"]

    # print("whole clip_f1 without last label: ")
    whole_label_seq_wo_last = sum([v[:-1] for v in whole_label_seq], [])
    whole_th_pred_seq_wo_last = sum([v[:-1] for v in whole_th_pred_seq], [])
    whole_th_f1 = round(f1_score(whole_label_seq_wo_last, whole_th_pred_seq_wo_last) * 100, 2)
    whole_th_precision = round(precision_score(whole_label_seq_wo_last, whole_th_pred_seq_wo_last) * 100, 2)
    whole_th_recall = round(recall_score(whole_label_seq_wo_last, whole_th_pred_seq_wo_last) * 100, 2)

    f1_tolerance_precision = th_f1_tolerance_hit / th_f1_tolerance_pred
    f1_tolerance_recall = th_f1_tolerance_hit / th_f1_tolerance_actual
    whole_th_f1_tolerance = 2 * f1_tolerance_precision * f1_tolerance_recall / (f1_tolerance_precision + f1_tolerance_recall)
    whole_th_f1_tolerance = round(whole_th_f1_tolerance * 100, 2)

    whole_th_bs = th_bs_hit / th_bs_total

    return th_bs_list, th_miou_by_overlap_list, whole_th_f1, whole_th_precision, whole_th_recall, whole_th_f1_tolerance, whole_th_bs, whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq


def compute_scores(pred_res, bs_metric_key="bs@30"):
    bs_threshold = int(bs_metric_key.split("@")[1])

    th_bs_list, th_miou_by_overlap_list, th_clip_f1_list, th_bs_v2_list = [], [], [], []
    topk_bs_list, topk_miou_by_overlap_list, topk_clip_f1_list, topk_bs_v2_list = [], [], [], []
    whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq = [], [], []
    th_bs_hit, th_bs_total, th_bs_hit_v2, th_bs_total_v2 = 0, 0, 0, 0
    topk_bs_hit, topk_bs_total, topk_bs_hit_v2, topk_bs_total_v2 = 0, 0, 0, 0

    th_f1_tolerance_hit, th_f1_tolerance_actual, th_f1_tolerance_pred = 0, 0, 0
    for index, example in enumerate(pred_res):
        example_id = example["example_id"]

        labels = example["labels"]
        whole_label_seq.append(labels)
        label_seconds = example["label_seconds"]

        threshold_preds = example["threshold_preds"]
        whole_th_pred_seq.append(threshold_preds)
        threshold_seconds = example["threshold_seconds"]
        threshold_seconds_v2 = example["threshold_seconds_v2"]

        topk_preds = example["topk_preds"]              # clip level 0 and 1 prediction
        whole_topk_pred_seq.append(topk_preds)
        topk_pred_seconds = example["topk_seconds"]     # end timestamp of each topic   
        topk_pred_seconds_v2 = example["topk_seconds_v2"]

        label_clusters = seconds2clusters(label_seconds)
        threshold_clusters = seconds2clusters(threshold_seconds)
        topk_clusters = seconds2clusters(topk_pred_seconds)

        label_seconds_from_clusters = list(np.cumsum(ecls_to_ts(label_clusters)))
        # print("label_seconds: {}, label_seconds_from_clusters: {}".format(label_seconds, label_seconds_from_clusters))
        # assert label_seconds == label_seconds_from_clusters

        th_score = get_score(label_clusters, threshold_clusters, labels, threshold_preds, threshold_seconds_v2, bs_threshold=bs_threshold)
        th_bs, th_miou, th_clip_f1 = th_score[bs_metric_key], th_score["miou"], th_score["clip_f1"]
        th_bs_list.append(th_bs)
        th_miou_by_overlap_list.append(th_miou)
        th_clip_f1_list.append(th_clip_f1)
        th_bs_v2_list.append(th_score["bs@{}v2".format(bs_threshold)])

        th_bs_hit += th_score["bs_hit"]
        th_bs_total += th_score["bs_total"]
        th_bs_hit_v2 += th_score["bs_hit_v2"]
        th_bs_total_v2 += th_score["bs_total_v2"]

        th_f1_tolerance_hit += th_score["f1_tolerance_hit"]
        th_f1_tolerance_actual += th_score["f1_tolerance_actual"]
        th_f1_tolerance_pred += th_score["f1_tolerance_pred"]

        topk_score = get_score(label_clusters, topk_clusters, labels, topk_preds, topk_pred_seconds_v2, bs_threshold=bs_threshold)
        topk_bs, topk_miou, topk_clip_f1 = topk_score[bs_metric_key], topk_score["miou"], topk_score["clip_f1"]
        topk_bs_list.append(topk_bs)
        topk_miou_by_overlap_list.append(topk_miou)
        topk_clip_f1_list.append(topk_clip_f1)
        topk_bs_v2_list.append(topk_score["bs@{}v2".format(bs_threshold)])

        topk_bs_hit += topk_score["bs_hit"]
        topk_bs_total += topk_score["bs_total"]
        topk_bs_hit_v2 += topk_score["bs_hit_v2"]
        topk_bs_total_v2 += topk_score["bs_total_v2"]

    # print("whole clip_f1 with last label: ")
    # print("threshold=0.5: ", f1_score(sum(whole_label_seq, []), sum(whole_th_pred_seq, [])))
    # print("topk_like_tt: ", f1_score(sum(whole_label_seq, []), sum(whole_topk_pred_seq, [])))

    # print("whole clip_f1 without last label: ")
    whole_label_seq_wo_last = sum([v[:-1] for v in whole_label_seq], [])
    whole_th_pred_seq_wo_last = sum([v[:-1] for v in whole_th_pred_seq], [])
    whole_topk_pred_seq_wo_last = sum([v[:-1] for v in whole_topk_pred_seq], [])
    whole_th_f1 = round(f1_score(whole_label_seq_wo_last, whole_th_pred_seq_wo_last) * 100, 2)
    whole_topk_f1 = round(f1_score(whole_label_seq_wo_last, whole_topk_pred_seq_wo_last) * 100, 2)

    f1_tolerance_precision = th_f1_tolerance_hit / th_f1_tolerance_pred
    f1_tolerance_recall = th_f1_tolerance_hit / th_f1_tolerance_actual
    whole_th_f1_tolerance = 2 * f1_tolerance_precision * f1_tolerance_recall / (f1_tolerance_precision + f1_tolerance_recall)
    whole_th_f1_tolerance = round(whole_th_f1_tolerance * 100, 2)

    whole_th_bs = th_bs_hit / th_bs_total
    whole_th_bs_v2 = th_bs_hit_v2 / th_bs_total_v2
    whole_topk_bs = topk_bs_hit / topk_bs_total
    whole_topk_bs_v2 = topk_bs_hit_v2 / topk_bs_total_v2

    # print("threshold=0.5: ", f1_score(whole_label_seq_wo_last, whole_th_pred_seq_wo_last))
    # print("topk_like_tt: ", f1_score(whole_label_seq_wo_last, whole_topk_pred_seq_wo_last))

    return th_bs_list, th_miou_by_overlap_list, th_clip_f1_list, whole_th_f1, whole_th_f1_tolerance, th_bs_v2_list, whole_th_bs, whole_th_bs_v2, \
        topk_bs_list, topk_miou_by_overlap_list, topk_clip_f1_list, whole_topk_f1, topk_bs_v2_list, whole_topk_bs, whole_topk_bs_v2, \
        whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq


def evaluate(data_file, pred_file, topk=-1, bs_threshold=30, logit_type="mm"):
    print("\n****evaluate****\n")
    bs_metric_key = "bs@{}".format(bs_threshold)
    f1_tolerance = "f1@{}".format(bs_threshold)
    pred_res = get_pred_result(data_file, pred_file, topk, logit_type)
    th_bs_list, th_miou_by_overlap_list, th_clip_f1_list, whole_th_f1, whole_th_f1_tolerance, th_bs_v2_list, whole_th_bs, whole_th_bs_v2, \
    topk_bs_list, topk_miou_by_overlap_list, topk_clip_f1_list, whole_topk_f1, topk_bs_v2_list, whole_topk_bs, whole_topk_bs_v2, \
     whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq = compute_scores(pred_res, bs_metric_key)

    example_num = len(th_bs_list)
    label_seq = sum([v[:-1] for v in whole_label_seq], [])
    th_pred_seq = sum([v[:-1] for v in whole_th_pred_seq], [])

    print("data_file: ", data_file)
    print("pred_file: ", pred_file)
    avg_label_num = round(sum(label_seq) / example_num, 2)
    avg_pred_num = round(sum(th_pred_seq) / example_num, 2)
    print("#example#pred/#label\n{}/{}/{}\n".format(example_num, avg_pred_num, avg_label_num))
    
    avg_th_bs = round(np.mean(np.array(th_bs_list)) * 100, 2)
    avg_th_miou_by_overlap = round(np.mean(np.array(th_miou_by_overlap_list)) * 100, 2)
    avg_th_clip_f1 = round(np.mean(np.array(th_clip_f1_list)) * 100, 2)
    avg_th_bs_v2 = round(np.mean(np.array(th_bs_v2_list)) * 100, 2)
    
    avg_topk_bs = round(np.mean(np.array(topk_bs_list)) * 100, 2)
    avg_topk_miou_by_overlap = round(np.mean(np.array(topk_miou_by_overlap_list)) * 100, 2)
    avg_topk_clip_f1 = round(np.mean(np.array(topk_clip_f1_list)) * 100, 2)
    avg_topk_bs_v2 = round(np.mean(np.array(topk_bs_v2_list)) * 100, 2)
    
    print("F1 / BS@{} / F1@{} / mIoU / avg".format(bs_threshold, bs_threshold))
    avg_th = round((whole_th_f1 + avg_th_bs + whole_th_f1_tolerance + avg_th_miou_by_overlap) / 4, 2)
    print("threshold=0.5: {} / {} / {} / {} / {}".format(whole_th_f1, avg_th_bs, whole_th_f1_tolerance, avg_th_miou_by_overlap, avg_th))
    
    return {
        "threshold": {
            "bs@{}".format(bs_threshold): avg_th_bs,
            "miou": avg_th_miou_by_overlap,
            # "clip_f1": avg_th_clip_f1,
            "whole_clip_f1": whole_th_f1,
            "bs@{}v2".format(bs_threshold): avg_th_bs_v2,
            f1_tolerance: whole_th_f1_tolerance,
        },
        "topk_like_tt": {
            "bs@{}".format(bs_threshold): avg_topk_bs,
            "miou": avg_topk_miou_by_overlap,
            # "clip_f1": avg_topk_clip_f1,
            "whole_clip_f1": whole_topk_f1,
            "bs@{}v2".format(bs_threshold): avg_topk_bs_v2,
        }
    }
    

def evaluate_by_type(data_file, pred_file, topk=-1, bs_threshold=30, language_type="en"):
    print("\n****evaluate_by_type****\n")
    bs_metric_key = "bs@{}".format(bs_threshold)
    pred_res = get_pred_result(data_file, pred_file, topk)
    th_bs_list, th_miou_by_overlap_list, th_clip_f1_list, whole_th_f1, whole_th_f1_tolerance, th_bs_v2_list, whole_th_bs, whole_th_bs_v2, \
    topk_bs_list, topk_miou_by_overlap_list, topk_clip_f1_list, whole_topk_f1, topk_bs_v2_list, whole_topk_bs, whole_topk_bs_v2, \
    whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq= compute_scores(pred_res, bs_metric_key)

    example_ids = [example_res["example_id"] for example_res in pred_res]
    def get_type_indices(type_, example_ids, language_type):
        if language_type == "en":
            type2video = en_type2video
            example_ids = [v.split("@@")[0] for v in example_ids]
        else:
            type2video = cn_type2video

        indices = []
        for index, course_id in enumerate(example_ids):
            if course_id in type2video[type_]:
                indices.append(index)
        return indices
    
    res = {}
    for type_ in types:
        type_indices = get_type_indices(type_, example_ids, language_type)

        type_th_bs = [th_bs_list[_] for _ in type_indices]
        type_th_miou = [th_miou_by_overlap_list[_] for _ in type_indices]
        type_th_clip_f1 = [th_clip_f1_list[_] for _ in type_indices]
        type_th_bs_v2 = [th_bs_v2_list[_] for _ in type_indices]

        type_topk_bs = [topk_bs_list[_] for _ in type_indices]
        type_topk_miou = [topk_miou_by_overlap_list[_] for _ in type_indices]
        type_topk_clip_f1 = [topk_clip_f1_list[_] for _ in type_indices]
        type_topk_bs_v2 = [topk_bs_v2_list[_] for _ in type_indices]

        type_label_seq = sum([whole_label_seq[_][:-1] for _ in type_indices], [])
        type_th_pred_seq = sum([whole_th_pred_seq[_][:-1] for _ in type_indices], [])
        type_topk_pred_seq = sum([whole_topk_pred_seq[_][:-1] for _ in type_indices], [])

        print("\n####type-{}####".format(type_))
        avg_th_bs = round(np.mean(np.array(type_th_bs)) * 100, 2)
        avg_th_bs_v2 = round(np.mean(np.array(type_th_bs_v2)) * 100, 2)
        avg_th_miou_by_overlap = round(np.mean(np.array(type_th_miou)) * 100, 2)
        whole_th_clip_f1 = round(f1_score(type_label_seq, type_th_pred_seq) * 100, 2)
        avg_th_clip_f1 = round(np.mean(np.array(type_th_clip_f1)) * 100, 2)
        
        avg_topk_bs = round(np.mean(np.array(type_topk_bs)) * 100, 2)
        avg_topk_bs_v2 = round(np.mean(np.array(type_topk_bs_v2)) * 100, 2)
        avg_topk_miou_by_overlap = round(np.mean(np.array(type_topk_miou)) * 100, 2)
        whole_topk_clip_f1 = round(f1_score(type_label_seq, type_topk_pred_seq) * 100, 2)
        avg_topk_clip_f1 = round(np.mean(np.array(type_topk_clip_f1)) * 100, 2)

        print("threshold0.5({} / whole_clip_f1): {} / {}\n".format(bs_metric_key, avg_th_bs, whole_th_clip_f1))

        print("threshold=0.5: {} / {} / {} / {}".format(avg_th_bs, avg_th_miou_by_overlap, whole_th_clip_f1, avg_th_clip_f1))
        print("topk_like_tt: {} / {} / {} / {}".format(avg_topk_bs, avg_topk_miou_by_overlap, whole_topk_clip_f1, avg_topk_clip_f1)) 
        print("topk_like_tt_bs/topk_like_tt_miou/threshold_whole_f1: {} / {} / {}".format(avg_topk_bs, avg_topk_miou_by_overlap, whole_th_clip_f1))

        # print("threshold=0.5: {} / {} / {} / {} / {}".format(whole_th_clip_f1, avg_th_bs_v2, avg_th_bs, avg_th_miou_by_overlap, avg_th_clip_f1))
        # print("topk_like_tt: {} / {} / {} / {} / {}".format(whole_topk_clip_f1, avg_topk_bs_v2, avg_topk_bs, avg_topk_miou_by_overlap, avg_topk_clip_f1)) 

        res[type_] = {
            "threshold": {
                bs_metric_key: avg_th_bs,
                "miou": avg_th_miou_by_overlap,
                "clip_f1": avg_th_clip_f1,
                "whole_clip_f1": whole_th_clip_f1,
                "bs@{}v2".format(bs_threshold): avg_th_bs_v2,
            },
            "topk_like_tt": {
                bs_metric_key: avg_topk_bs,
                "miou": avg_topk_miou_by_overlap,
                "clip_f1": avg_topk_clip_f1,
                "whole_clip_f1": whole_topk_clip_f1,
                "bs@{}v2".format(bs_threshold): avg_topk_bs_v2,
            }
        }
    
    return res


def evaluate_multi_experimens(data_file, pred_files, bs_threshold=30, language_type="en"):
    print("\n****evaluate_multi_experimens****\n")
    bs_metric_key = "bs@{}".format(bs_threshold)
    f1_tolerance = "f1@{}".format(bs_threshold)
    miou_metric_key = "miou"
    clip_f1_metric_key = "whole_clip_f1"
    topk_key, threshold_key = "topk_like_tt", "threshold"

    all_topk_bs, all_topk_bs_v2, all_topk_miou, all_topk_clip_f1 = [], [], [], []
    type_topk_bs, type_topk_bs_v2, type_topk_miou, type_topk_clip_f1 = {}, {}, {}, {}

    all_threshold_bs, all_threshold_bs_v2, all_threshold_miou, all_threshold_clip_f1, all_threshold_f1_tolerance = [], [], [], [], []
    type_threshold_bs, type_threshold_bs_v2, type_threshold_miou, type_threshold_clip_f1, type_threshold_f1_tolerance = {}, {}, {}, {}, {}

    for pred_file in pred_files:
        all_res = evaluate(data_file, pred_file, bs_threshold=bs_threshold)
        all_topk_bs.append(all_res[topk_key][bs_metric_key])
        all_topk_bs_v2.append(all_res[topk_key]["bs@{}v2".format(bs_threshold)])
        all_topk_miou.append(all_res[topk_key][miou_metric_key])
        all_topk_clip_f1.append(all_res[topk_key][clip_f1_metric_key])

        all_threshold_bs.append(all_res[threshold_key][bs_metric_key])
        all_threshold_bs_v2.append(all_res[threshold_key]["bs@{}v2".format(bs_threshold)])
        all_threshold_miou.append(all_res[threshold_key][miou_metric_key])
        all_threshold_clip_f1.append(all_res[threshold_key][clip_f1_metric_key])
        all_threshold_f1_tolerance.append(all_res[threshold_key][f1_tolerance])
        
        type_res = evaluate_by_type(data_file, pred_file, bs_threshold=bs_threshold, language_type=language_type)
        for type_ in type_res:
            if type_ not in type_topk_bs:
                type_topk_bs[type_] = []
                type_topk_bs_v2[type_] = []
                type_topk_miou[type_] = []
                type_topk_clip_f1[type_] = []
                type_threshold_bs[type_] = []
                type_threshold_bs_v2[type_] = []
                type_threshold_miou[type_] = []
                type_threshold_clip_f1[type_] = []
            type_topk_bs[type_].append(type_res[type_][topk_key][bs_metric_key])
            type_topk_bs_v2[type_].append(type_res[type_][topk_key]["bs@{}v2".format(bs_threshold)])
            type_topk_miou[type_].append(type_res[type_][topk_key][miou_metric_key])
            type_topk_clip_f1[type_].append(type_res[type_][topk_key][clip_f1_metric_key])

            type_threshold_bs[type_].append(type_res[type_][threshold_key][bs_metric_key])
            type_threshold_bs_v2[type_].append(type_res[type_][threshold_key]["bs@{}v2".format(bs_threshold)])
            type_threshold_miou[type_].append(type_res[type_][threshold_key][miou_metric_key])
            type_threshold_clip_f1[type_].append(type_res[type_][threshold_key][clip_f1_metric_key])

    # print("\n\n########{}########".format(topk_key))
    # print("\nall-avg {}/{}/{}".format(bs_metric_key, miou_metric_key, clip_f1_metric_key))
    # print("\n".join(["{} / {} / {}".format(b, c, d) for a, b, c, d in zip(all_topk_bs_v2, all_topk_bs, all_topk_miou, all_topk_clip_f1)]))
    # print("{}({}) / {}({}) / {}({})".format(
    #     # round(np.mean(np.array(all_topk_bs_v2)), 2), round(statistics.stdev(all_topk_bs_v2), 2),
    #     round(np.mean(np.array(all_topk_bs)), 2), round(statistics.stdev(all_topk_bs), 2),
    #     round(np.mean(np.array(all_topk_miou)), 2), round(statistics.stdev(all_topk_miou), 2), 
    #     round(np.mean(np.array(all_topk_clip_f1)), 2), round(statistics.stdev(all_topk_clip_f1), 2),
    #     ))

    # for type_ in types:
    #     print("\n{}-avg {}/{}/{}".format(type_, bs_metric_key, miou_metric_key, clip_f1_metric_key))
    #     print("\n".join(["{} / {} / {}".format(b, c, d) for a, b, c, d in zip(type_topk_bs_v2[type_], type_topk_bs[type_], type_topk_miou[type_], type_topk_clip_f1[type_])]))
    #     print("{}({}) / {}({}) / {}({})".format(
    #         # round(np.mean(np.array(type_topk_bs_v2[type_])), 2), round(statistics.stdev(type_topk_bs_v2[type_]), 2),
    #         round(np.mean(np.array(type_topk_bs[type_])), 2), round(statistics.stdev(type_topk_bs[type_]), 2), 
    #         round(np.mean(np.array(type_topk_miou[type_])), 2), round(statistics.stdev(type_topk_miou[type_]), 2), 
    #         round(np.mean(np.array(type_topk_clip_f1[type_])), 2), round(statistics.stdev(type_topk_clip_f1[type_]), 2),
    #         ))

    print("\n\n########{}########".format(threshold_key))
    print("\nall-avg {}/{}/{}/{}".format(bs_metric_key, miou_metric_key, clip_f1_metric_key, f1_tolerance))
    print("\n".join(["{} / {} / {} / {}".format(b, c, d, e) for a, b, c, d, e in zip(all_threshold_bs_v2, all_threshold_bs, all_threshold_miou, all_threshold_clip_f1, all_threshold_f1_tolerance)]))
    print("{}({}) / {}({}) / {}({}) / {}({})".format(
        # round(np.mean(np.array(all_threshold_bs_v2)), 2), round(statistics.stdev(all_threshold_bs_v2), 2),
        round(np.mean(np.array(all_threshold_bs)), 2), round(statistics.stdev(all_threshold_bs), 2),
        round(np.mean(np.array(all_threshold_miou)), 2), round(statistics.stdev(all_threshold_miou), 2), 
        round(np.mean(np.array(all_threshold_clip_f1)), 2), round(statistics.stdev(all_threshold_clip_f1), 2),
        round(np.mean(np.array(all_threshold_f1_tolerance)), 2), round(statistics.stdev(all_threshold_f1_tolerance), 2),
        ))

    # for type_ in types:
    #     print("\n{}-avg {}/{}/{}".format(type_, bs_metric_key, miou_metric_key, clip_f1_metric_key))
    #     print("\n".join(["{} / {} / {}".format(b, c, d, e) for a, b, c, d, e in zip(type_threshold_bs_v2[type_], type_threshold_bs[type_], type_threshold_miou[type_], type_threshold_clip_f1[type_], type_threshold_f1_tolerance)]))
    #     print("{}({}) / {}({}) / {}({})".format(
    #         # round(np.mean(np.array(type_threshold_bs_v2[type_])), 2), round(statistics.stdev(type_threshold_bs_v2[type_]), 2),
    #         round(np.mean(np.array(type_threshold_bs[type_])), 2), round(statistics.stdev(type_threshold_bs[type_]), 2), 
    #         round(np.mean(np.array(type_threshold_miou[type_])), 2), round(statistics.stdev(type_threshold_miou[type_]), 2), 
    #         round(np.mean(np.array(type_threshold_clip_f1[type_])), 2), round(statistics.stdev(type_threshold_clip_f1[type_]), 2),
    #         ))


def evaluate_llm(data_file, pred_file, bs_threshold=30):
    print("\n****evaluate****\n")
    print("data_file: ", data_file)
    print("pred_file: ", pred_file)

    bs_metric_key = "bs@{}".format(bs_threshold)
    f1_tolerance = "f1@{}".format(bs_threshold)
    pred_res = get_llm_result(data_file, pred_file)
    th_bs_list, th_miou_by_overlap_list, whole_th_f1, whole_th_p, whole_th_r, whole_th_f1_tolerance, whole_th_bs, whole_label_seq, whole_th_pred_seq, whole_topk_pred_seq = compute_scores_llm(pred_res, bs_metric_key)
    
    avg_th_bs = round(np.mean(np.array(th_bs_list)) * 100, 2)
    avg_th_miou_by_overlap = round(np.mean(np.array(th_miou_by_overlap_list)) * 100, 2)
    
    print("bs@{} / mIoU / whole_clip_f1 / precision / recall / f1@{}".format(bs_threshold, bs_threshold))
    print("{} / {} / {} / {} / {} / {}".format(avg_th_bs, avg_th_miou_by_overlap, whole_th_f1, whole_th_p, whole_th_r, whole_th_f1_tolerance))

    print("avg_pred / avg_true / examples")
    boundary_label_num = sum([sum(example["labels"]) - 1 for example in pred_res])      # remove last boundary
    boundary_pred_num = sum([sum(example["threshold_preds"]) - 1 for example in pred_res])
    print("{} / {} / {}".format(round(boundary_pred_num / len(pred_res), 2), round(boundary_label_num / len(pred_res), 2), len(pred_res)))


def evaluate_vstar(data_file, pred_file):
    # pk / wd / macro-f1
    # 参考：https://github.com/lxing532/Dialogue-Topic-Segmenter/blob/main/segment.py
    # 样本结果取平均，且包含最后一个边界

    from segeval.window.pk import pk as PK
    from segeval.window.windowdiff import window_diff as WD

    def mass_from_end_label_sequence(labels):
        # if labels[i] == 1, then i_th sentence is the end sentence of its paragraph
        # [1, 1, 0, 0, 1, 1] -> [1, 1, 3, 1]
        mass = []
        cur_cnt = 0
        for v in labels:
            if v == 1:
                cur_cnt += 1
                mass.append(cur_cnt)
                cur_cnt = 0
            else:
                cur_cnt += 1
        if cur_cnt > 0:
            mass.append(cur_cnt)
        return mass
    
    def compute_metric_from_mass_data(hypothesis, reference):
        # input's format is 1-d mass
        pk = PK(hypothesis, reference)
        wd = WD(hypothesis, reference)
        return pk, wd
    
    # get result
    prediction_key = "predictions"
    all_data = read_jsonl_file(data_file)
    all_pred = read_jsonl_file(pred_file)

    total_pk, total_wd, total_macro_f1, num_example = 0, 0, 0, 0
    whole_labels, whole_preds = [], []
    for example_info, pred_info in zip(all_data, all_pred):
        num_example += 1
        labels = example_info["labels"][:-1]        # remove last boundary temporarily
        predictions = pred_info[prediction_key][:len(labels)]
        whole_labels += labels
        whole_preds += predictions

        labels.append(1)
        predictions.append(1)
        macro_f1 = f1_score(labels, predictions, labels = [0,1], average='macro')

        label_mass = mass_from_end_label_sequence(labels)
        pred_mass = mass_from_end_label_sequence(predictions)
        pk, wd = compute_metric_from_mass_data(pred_mass, label_mass)

        total_pk += pk
        total_wd += wd
        total_macro_f1 += macro_f1

    avg_pk = round(total_pk / num_example, 2)
    avg_wd = round(total_wd / num_example, 2)
    avg_macro_f1 = round(total_macro_f1 / num_example, 2)
    pos_f1 = round(f1_score(whole_labels, whole_preds), 2)
    print("pk / wd / macro_f1 / pos_f1")
    print("{} / {} / {} / {}".format())


if __name__ == "__main__":
    # en
    pred_file = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/code/mmts/text_with_visual/output/longformer_base-finetune-avlecture/seed42-fuse_cat-pfcn1-vis_True-ocr_True-mcl_False_0.5-ttcl_True_0.5-tvcl_False_0.5-seq2048-wlz0.7-lr5e-5-epoch5-bs8-2024-03-20_14:13:29/predict_avl_mm_clip_level_0308_max_seq2048.txt"
    pred_file = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/code/mmts/swst_like_v2/output/t_longformer_base-v2d_clip_vit_b_16-vis_before_bos_True-vatFalse-vtn3-vfn1-p_linear_l1_moe4_top2_lblw0.1-finetune-avlecture-seq2048-wlz0.6-lr5e-5-epoch5-bs8/seed42-fuse_cat-vis2d_True-vis3d_True-ocr_True/mcl_False_0.5-ttcl_False_0.1-tvcl_False_0.1-cl_list_pos1_neg4_random-2024-04-11_21:28:58/predict_avl_mm_clip_level_0308_max_seq2048.txt"
    
    #### 多次实验求平均
    pred_file_name = "predict_avl_mm_clip_level_0308_max_seq2048.txt"
    # text baseline
    pred_folders = [
        "/mnt/workspace/workgroup/yuhai/lecture_video_seg/code/mmts/text_with_visual/output/longformer_base-finetune-avlecture/seed42-vis_False-ocr_False-seq2048-wlz0.7-lr5e-5-epoch5-bs8-2024-03-12_19:37:51",
        "/mnt/workspace/workgroup/yuhai/lecture_video_seg/code/mmts/text_with_visual/output/longformer_base-finetune-avlecture/seed59-pfcn1-vis_False-ocr_False-seq2048-wlz0.7-lr5e-5-epoch5-bs8-2024-03-13_18:06:52",
        "/mnt/workspace/workgroup/yuhai/lecture_video_seg/code/mmts/text_with_visual/output/longformer_base-finetune-avlecture/seed88-pfcn1-vis_False-ocr_False-seq2048-wlz0.7-lr5e-5-epoch5-bs8-2024-03-13_21:44:19",
    ]

    # cn
    #### 多次实验求平均
    pred_file_name = "predict_cn_course_0604_max_seq2048.txt"

    parser = argparse.ArgumentParser(description='run evaluate.py')
    parser.add_argument('-d', '--data_file', type=str, help='data file path', default="")
    parser.add_argument('-p', '--pred_file', type=str, help='pred file path', default=pred_file)
    parser.add_argument('-a', '--avg', type=bool, help='compute average metric', default=False)
    parser.add_argument('-pfs', '--pred_folders', type=list, help='compute average metric', default=pred_folders)
    parser.add_argument("-bs", "--bs_threshold", type=int, help="bs_threshold", default=30)
    parser.add_argument("-l", "--language_type", type=str, help="cn or en", default="en")
    parser.add_argument("-type", "--type", type=str, default="lvts")
    parser.add_argument("-logit", "--logit_type", type=str, default="mm")  # choices are mm, text, vis
    
    args = parser.parse_args()
    if args.data_file == "":
        args.data_file = cn_data_file if args.language_type == "cn" else en_data_file
    
    if args.avg:
        # TODO in bash arguments
        print("compute average metric.")
        assert args.type != "llm"
        pred_files = [os.path.join(pred_folder, pred_file_name) for pred_folder in args.pred_folders]
        evaluate_multi_experimens(data_file, pred_files, bs_threshold=args.bs_threshold, language_type=args.language_type)
    else:
        if args.type == "llm":
            evaluate_llm(args.data_file, args.pred_file, bs_threshold=args.bs_threshold)
        elif args.type == "vstar":
            evaluate_vstar(args.data_file, args.pred_file)
        else:
            evaluate(args.data_file, args.pred_file, bs_threshold=args.bs_threshold, logit_type=args.logit_type)
            # evaluate_by_type(args.data_file, args.pred_file, bs_threshold=args.bs_threshold, language_type=args.language_type)
