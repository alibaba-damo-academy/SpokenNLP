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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from segeval.window.pk import pk as PK
from segeval.window.windowdiff import window_diff as WD
import numpy as np
from tokenizer import BasicTokenizer
from config import MS_SDK_TOKEN
from rouge import Rouge as rouge_scorer
import sys
import pandas as pd
from copy import deepcopy
from urllib import request
import ssl
from rouge import Rouge


sys.setrecursionlimit(100000)

bTokenizer = BasicTokenizer()
tokenize_func = bTokenizer.tokenize


def read_jsonl_local(input_file):
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def read_online(split, subset_name="default", num_lines=None):
    assert split == "train" or split == "validation"
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
        subset_name=subset_name,
        **input_config_kwargs)
    if num_lines:
        data = data[split].select(range(num_lines))
    else:
        data = data[split]
    sample = data.map(lambda x: json.loads(x['content']), batched=False, remove_columns=["idx", "content"])
    return sample


# read_jsonl = read_jsonl_online
read_jsonl = read_jsonl_local


def compute_window_metric(predictions, references, prefix):
    def mass_from_start_label_sequence(labels):
        # if labels[i] == 1, then i_th sentence is the start sentence of its paragraph
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

    n = len(predictions)
    case_results = {
        "1-pk": [],
        "1-wd": [],
    }
    for i, (y_pred_label, y_true_label) in enumerate(zip(predictions, references)):
        try:
            y_pred_mass = mass_from_start_label_sequence(y_pred_label)
            y_true_mass = mass_from_start_label_sequence(y_true_label)
            assert sum(y_pred_mass) == sum(y_true_mass)
            pk, wd = compute_metric_from_mass_data(y_pred_mass, y_true_mass)
            case_results["1-pk"].append(1 - pk)
            case_results["1-wd"].append(1 - wd)
        except Exception as e:
            print(i, e)
            raise RuntimeError
    total_result = {
        "1-pk": round(float(np.array(case_results["1-pk"]).mean()), 4),
        "1-wd": round(float(np.array(case_results["1-wd"]).mean()), 4),
    }

    predictions = sum(predictions, [])  # https://www.zhihu.com/question/269103728
    references = sum(references, [])
    p = precision_score(references, predictions)
    r = recall_score(references, predictions)
    f1 = f1_score(references, predictions)
    micro_f1 = f1_score(references, predictions, average='micro')

    avg_pred_cnt = round(sum(predictions) * 1.0 / n, 2)
    avg_true_cnt = round(sum(references) * 1.0 / n, 2)
    return {
        prefix + "1-pk": total_result["1-pk"],
        prefix + "1-wd": total_result["1-wd"],
        prefix + "precision": round(p, 4),
        prefix + "recall": round(r, 4),
        prefix + "f1": round(f1, 4),
        prefix + "avg_pred_cnt": avg_pred_cnt,
        prefix + "avg_true_cnt": avg_true_cnt,
    }


def topic_segment_score_func(pos_f1, one_minus_pk, one_minus_wd):
    score = 0.5 * pos_f1 + 0.25 * (one_minus_pk + one_minus_wd)
    return score


def evaluate_multi_files(split, input_pred_files, task):
    total_res = []
    for pred_file in input_pred_files:
        pred_file = deepcopy(pred_file)
        if task == "topic_segmentation":
            eval_res = topic_segment_evaluate(split, input_pred_file=pred_file["result"])
        elif task == "extractive_summarization":
            eval_res = extractive_summarization_evaluate(split, input_pred_file=pred_file["result"])
        elif task == "topic_title_generation":
            eval_res = topic_title_generation_evaluate(split, input_pred_file=pred_file["result"])
        elif task == "keyphrase_extraction":
            eval_res = keyphrase_extraction_evaluate(split, input_pred_file=pred_file["result"])
        elif task == "action_item_detection":
            eval_res = action_item_detection_evaluate(split, input_pred_file=pred_file["result"])
        else:
            raise NotImplementedError
        del pred_file["result"]
        pred_file.update(eval_res)
        total_res.append(pred_file)
    df = pd.DataFrame.from_records(total_res)
    csv_string = df.to_csv()
    return csv_string


def topic_segment_evaluate(split, input_pred_file):
    pred_samples = read_jsonl(input_pred_file)
    label_samples = read_online(split, num_lines=len(pred_samples))
    assert label_samples.num_rows == len(pred_samples), "NUMBER ERROR."
    total_preds = []
    total_labels = []
    total_preds_split = []
    total_labels_split = []

    for l_sample, p_sample in zip(label_samples, pred_samples):
        assert l_sample["meeting_key"] == p_sample["meeting_key"], "meeting_key error."
        sentences = l_sample["sentences"]
        para_segment_ids = [_['id'] for _ in l_sample["paragraph_segment_ids"]]
        preds = [0] * len(sentences)
        labels = [0] * len(sentences)
        label_pos_ids = [_["id"] for _ in l_sample["topic_segment_ids"]]
        pred_pos_ids = [_["id"] for _ in p_sample["topic_segment_ids"]]

        for pos_id in label_pos_ids:
            labels[pos_id-1] = 1
        for pos_id in pred_pos_ids:
            preds[pos_id-1] = 1

        preds[-1] = 1
        labels[-1] = 1

        labels = [_ for sent_index, _ in enumerate(labels) if (sent_index+1) in para_segment_ids]
        preds = [_ for sent_index, _ in enumerate(preds) if (sent_index+1) in para_segment_ids]

        total_labels.extend(labels[:-1])
        total_preds.extend(preds[:-1])

        total_labels_split.append(labels[:-1])
        total_preds_split.append(preds[:-1])

    f1_score = classification_report(y_true=total_labels, y_pred=total_preds, output_dict=True)
    window_score = compute_window_metric(predictions=total_preds_split, references=total_labels_split, prefix="test_")
    rank_score = topic_segment_score_func(pos_f1=f1_score["1"]['f1-score'], one_minus_pk=window_score["test_1-pk"], one_minus_wd=window_score["test_1-wd"])
    del window_score["test_avg_pred_cnt"]
    del window_score["test_avg_true_cnt"]
    out_dict = {'score': rank_score}
    out_dict.update(window_score)
    return out_dict


def rouge_compute(predictions, references, use_avg=True):
    scorer = rouge_scorer()
    # print("pred", predictions)
    # print("ref", references)

    predictions = [" ".join(tokenize_func(_)) for _ in predictions]
    references = [" ".join(tokenize_func(_)) for _ in references]
    # print(predictions)
    # print(references)
    scores = scorer.get_scores(predictions, references, avg=use_avg)

    result = {"score": scores["rouge-1"]["f"]}
    for key1 in scores.keys():
        for key2 in scores[key1]:
            result["{}_{}".format(key1, key2)] = scores[key1][key2]

    return result


def compute_es_rouge(total_es_label, total_es_pred):
    multi_rouge_scores_average = []
    multi_rouge_scores_max = []

    for pred, multi_ref in zip(total_es_pred, total_es_label):
        # print(pred)
        assert isinstance(multi_ref, list)
        # print("multi_ref", multi_ref)
        multi_rouge = [rouge_compute([pred], [_]) for _ in multi_ref]
        max_score = max(multi_rouge, key=lambda x: x["rouge-l_f"])
        multi_rouge_scores_max.append(max_score)

        ave_score = {}
        for key in max_score.keys():
            ave = np.mean([_[key] for _ in multi_rouge])
            ave_score[key] = ave
        multi_rouge_scores_average.append(ave_score)

    multi_ref_ave_rouge = {}
    multi_ref_max_rouge = {}
    for key in max_score.keys():
        ave = np.mean([_[key] for _ in multi_rouge_scores_average])
        multi_ref_ave_rouge["multi-ref-average_{}".format(key)] = ave

        ave = np.mean([_[key] for _ in multi_rouge_scores_max])
        multi_ref_max_rouge["multi-ref-max_{}".format(key)] = ave

    out = {}
    out.update(multi_ref_max_rouge)
    out.update(multi_ref_ave_rouge)
    return out


def extractive_summarization_score_func(mean_items):
    assert len(mean_items) == 12
    score = np.mean(mean_items)
    return score


def extractive_summarization_evaluate(split, input_pred_file):
    pred_samples = read_jsonl(input_pred_file)
    label_samples = read_online(split, num_lines=len(pred_samples))
    assert len(label_samples) == len(pred_samples), "NUMBER ERROR."

    total_topic_es_label = []
    total_topic_es_pred = []
    total_doc_es_label = []
    total_doc_es_pred = []
    for l_sample, p_sample in zip(label_samples, pred_samples):
        assert l_sample["meeting_key"] == p_sample["meeting_key"], "meeting_key error."
        sentences = [_['s'] for _ in l_sample["sentences"]]
        label_topics = l_sample["topic_segment_ids"]
        pred_topics = p_sample["topic_segment_ids"]
        assert len(label_topics) == len(pred_topics)

        topic_es_multi_ref = []
        for topic in label_topics:
            refs = []
            for ref in topic["candidate"]:
                ref_summarization = "".join([sentences[int(sent_index)-1] for sent_index in ref["key_sentence"]])
                refs.append(ref_summarization)
            topic_es_multi_ref.append(refs)

        topic_es_pred = []
        for topic in pred_topics:
            pred_summarization = "".join([sentences[int(sent_index)-1] for sent_index in topic["key_sentence"]])
            topic_es_pred.append(pred_summarization)

        total_topic_es_label.extend(topic_es_multi_ref)
        total_topic_es_pred.extend(topic_es_pred)

        doc_refs = []
        for ref in l_sample["candidate"]:
            ref_summarization = "".join([sentences[int(sent_index) - 1] for sent_index in ref["key_sentence"]])
            doc_refs.append(ref_summarization)
        total_doc_es_label.append(doc_refs)

        doc_pred_summarization = "".join([sentences[int(sent_index) - 1] for sent_index in p_sample["key_sentence"]])
        total_doc_es_pred.append(doc_pred_summarization)

    # print(total_topic_es_pred)
    # print(total_topic_es_label)
    for sss in total_topic_es_label:
        assert isinstance(sss, list)
    for sss in total_doc_es_label:
        assert isinstance(sss, list)

    assert len(total_topic_es_label) == len(total_topic_es_pred)

    topic_es_res = compute_es_rouge(total_es_label=total_topic_es_label, total_es_pred=total_topic_es_pred)
    doc_es_res = compute_es_rouge(total_es_label=total_doc_es_label, total_es_pred=total_doc_es_pred)

    # print(topic_es_res)
    # print(doc_es_res)
    score_items = []
    for es_res in [topic_es_res, doc_es_res]:
        for s_type in ["average", "max"]:
            for s_value in ["1", '2', 'l']:
                score_items.append(es_res["multi-ref-{}_rouge-{}_f".format(s_type, s_value)])
    # print(score_items)
    rank_score = extractive_summarization_score_func(score_items)

    out_dict = {'score': rank_score, "extra_metrics": {}}
    for es_name, es_res in zip(["topic-es_", "doc-es_"], [topic_es_res, doc_es_res]):
        for key in es_res.keys():
            new_key = key.replace("multi-ref-", es_name)
            out_dict["extra_metrics"][new_key] = es_res[key]
    out_dict.update(out_dict["extra_metrics"])
    del out_dict["extra_metrics"]
    return out_dict


def topic_title_generation_score_func(mean_items):
    assert len(mean_items) == 6
    score = np.mean(mean_items)
    return score


def topic_title_generation_evaluate(split, input_pred_file):
    pred_samples = read_jsonl(input_pred_file)
    label_samples = read_online(split, num_lines=len(pred_samples))
    assert len(label_samples) == len(pred_samples), "NUMBER ERROR."

    total_topic_ttg_label = []
    total_topic_ttg_pred = []
    for l_sample, p_sample in zip(label_samples, pred_samples):
        assert l_sample["meeting_key"] == p_sample["meeting_key"], "meeting_key error."
        sentences = [_['s'] for _ in l_sample["sentences"]]
        label_topics = l_sample["topic_segment_ids"]
        pred_topics = p_sample["topic_segment_ids"]
        assert len(label_topics) == len(pred_topics)

        topic_ttg_multi_ref = []
        for topic in label_topics:
            refs = [ref["title"] for ref in topic["candidate"]]
            topic_ttg_multi_ref.append(refs)

        topic_ttg_pred = []
        for topic in pred_topics:
            topic_ttg_pred.append(topic["title"])

        total_topic_ttg_label.extend(topic_ttg_multi_ref)
        total_topic_ttg_pred.extend(topic_ttg_pred)

    for sss in total_topic_ttg_label:
        assert isinstance(sss, list)

    assert len(total_topic_ttg_label) == len(total_topic_ttg_pred)

    topic_ttg_res = compute_es_rouge(total_es_label=total_topic_ttg_label, total_es_pred=total_topic_ttg_pred)

    score_items = []
    for es_res in [topic_ttg_res]:
        for s_type in ["average", "max"]:
            for s_value in ["1", '2', 'l']:
                score_items.append(es_res["multi-ref-{}_rouge-{}_f".format(s_type, s_value)])
    # print(score_items)
    rank_score = topic_title_generation_score_func(score_items)

    out_dict = {'score': rank_score, "extra_metrics": {}}
    for task_name, es_res in zip(["ttg_"], [topic_ttg_res]):
        for key in es_res.keys():
            new_key = key.replace("multi-ref-", task_name)
            out_dict["extra_metrics"][new_key] = es_res[key]
    out_dict.update(out_dict["extra_metrics"])
    del out_dict["extra_metrics"]
    return out_dict


def kpe_compute(predictions, references):
    scores = {}
    score_sum = 0.0
    for num in [10, 15, 20]:
        predictions_at_num = [pred[:num] for pred in predictions]
        approximate_match_score = calculateCorpusApproximateMatchScore(predictions_at_num, references)
        rouge_score = calculateRouge(predictions_at_num, references)

        for k, v in approximate_match_score.items():
            scores[k + "@%d" % num] = v
            score_sum += v
        for k, v in rouge_score.items():
            scores[k + "@%d" % num] = v
            score_sum += v
    out_dict = {'score': score_sum / len(scores.keys())}
    out_dict.update(scores)
    return out_dict


def calculateCorpusApproximateMatchScore(keywords, goldenwords):
    # print("calculateCorpusApproximateMatchScore...")
    partial_f1_list = []
    for example_keywords, example_goldenwords in zip(keywords, goldenwords):
        example_score = calculateExampleApproximateMatchScore(example_keywords, example_goldenwords)
        partial_f1_list.append(example_score["partial_f1"])

    partial_f1 = sum(partial_f1_list) * 1.0 / len(partial_f1_list)
    return {"partial_f1": partial_f1}


def calculateExampleApproximateMatchScore(keywords, goldenwords):

    def isFuzzyMatch(firststring, secondstring):
        # 判断两个字符串是否模糊匹配;标准是最长公共子串长度是否>=2
        first_str = firststring.strip()
        second_str = secondstring.strip()
        len_1 = len(first_str)
        len_2 = len(second_str)
        len_vv = []
        global_max = 0
        for i in range(len_1 + 1):
            len_vv.append([0] * (len_2 + 1))
        if len_1 == 0 or len_2 == 0:
            return False
        for i in range(1, len_1 + 1):
            for j in range(1, len_2 + 1):
                if first_str[i - 1] == second_str[j - 1]:  # 根据对应的字符是否相等来判断
                    len_vv[i][j] = 1 + len_vv[i - 1][j - 1]  # 长度二维数组的值
                else:
                    len_vv[i][j] = 0
                global_max = max(global_max, len_vv[i][j])
        if global_max >= 2:
            return True
        return False

    recallLength = len(goldenwords)
    precisionLength = len(keywords)
    nonRecall = []
    precisionNum = 0
    recallNum = 0
    for _key in keywords:
        for _goldenkey in goldenwords:
            if isFuzzyMatch(_key, _goldenkey):
                precisionNum += 1
                break
    for _goldenkey in goldenwords:
        flag = False
        for _key in keywords:
            if isFuzzyMatch(_key, _goldenkey):
                flag = True
                recallNum += 1
                break
        if not flag:
            nonRecall.append(_goldenkey)
    precisionScore = float(precisionNum) / float(precisionLength)
    recallScore = float(recallNum) / float(recallLength)
    # print(precisionScore, recallScore)
    # print(precisionNum, precisionLength)
    # print(recallNum, recallLength)
    fScore = 0.0
    if ((precisionScore + recallScore) != 0):
        fScore = 2 * precisionScore * recallScore / (precisionScore + recallScore)
    return {
        'partial_f1': fScore,
    }


def calculateRouge(keywords, goldenwords):
    keywords = [" ".join(_) for _ in keywords]
    goldenwords = [" ".join(_) for _ in goldenwords]
    # print("keywords: ", keywords)
    # print("goldenwords: ", goldenwords)
    rouge = Rouge()
    scores = rouge.get_scores(hyps=keywords, refs=goldenwords, avg=True)
    return {
        'exact_f1': scores['rouge-1']['f'],
    }


def keyphrase_extraction_evaluate(split, input_pred_file):
    pred_samples = read_jsonl(input_pred_file)
    label_samples = read_online(split, num_lines=len(pred_samples))
    assert len(label_samples) == len(pred_samples), "NUMBER ERROR."

    total_preds = []
    total_labels = []

    for l_sample, p_sample in zip(label_samples, pred_samples):
        assert l_sample["meeting_key"] == p_sample["meeting_key"], "meeting_key error."
        key_word = [c["key_word"] for c in l_sample["candidate"]]
        key_word = [w for ww in key_word for w in ww]
        label_kp = key_word
        pred_kp = p_sample["key_word"]
        total_preds.append(pred_kp)
        total_labels.append(label_kp)

    out_res = kpe_compute(predictions=total_preds, references=total_labels)
    return out_res


def action_item_detection_evaluate(split, input_pred_file):
    pred_samples = read_jsonl(input_pred_file)
    label_samples = read_online(split, num_lines=len(pred_samples))
    assert len(label_samples) == len(pred_samples), "NUMBER ERROR."

    total_preds = []
    total_labels = []

    for l_sample, p_sample in zip(label_samples, pred_samples):
        assert l_sample["meeting_key"] == p_sample["meeting_key"], "meeting_key error."
        sentences = l_sample["sentences"]
        preds = [0] * len(sentences)
        labels = [0] * len(sentences)
        label_pos_ids = [_["id"] for _ in l_sample["action_ids"]]
        pred_pos_ids = [_["id"] for _ in p_sample["action_ids"]]

        for pos_id in label_pos_ids:
            labels[pos_id - 1] = 1
        for pos_id in pred_pos_ids:
            preds[pos_id - 1] = 1

        total_labels.extend(labels)
        total_preds.extend(preds)

    f1_score = classification_report(y_true=total_labels, y_pred=total_preds, output_dict=True)['1']
    del f1_score['support']
    out_dict = {"score": f1_score["f1-score"]}
    out_dict.update(f1_score)
    return out_dict


if __name__ == "__main__":
    split = "validation" # validation or train
    pred_file = "./../../submitted_samples/topic_segmentation_dev_pesudo_submit.jsonl"
    pred_files = [{"recodId": 1,"result":pred_file}, {"recodId": 2,"result":pred_file}]

    out_res = evaluate_multi_files(split=split, input_pred_files=pred_files, task="topic_segmentation")
    print(out_res)

    pred_file = "./../../submitted_samples/extractive_summarization_dev_pesudo_submit.jsonl"
    pred_files = [{"recodId": 1,"result":pred_file}, {"recodId": 2,"result":pred_file}]

    out_res = evaluate_multi_files(split=split, input_pred_files=pred_files, task="extractive_summarization")
    print(out_res)

    pred_file = "./../../submitted_samples/topic_title_generation_dev_pesudo_submit.jsonl"
    pred_files = [{"recodId": 1,"result":pred_file}, {"recodId": 2,"result":pred_file}]

    out_res = evaluate_multi_files(split=split, input_pred_files=pred_files, task="topic_title_generation")
    print(out_res)

    pred_file = "./../../submitted_samples/keyphrase_extraction_dev_pesudo_submit.jsonl"
    pred_files = [{"recodId": 1,"result":pred_file}, {"recodId": 2,"result":pred_file}]

    out_res = evaluate_multi_files(split=split, input_pred_files=pred_files, task="keyphrase_extraction")
    print(out_res)

    pred_file = "./../../submitted_samples/action_item_detection_dev_pesudo_submit.jsonl"
    pred_files = [{"recodId": 1,"result":pred_file}, {"recodId": 2,"result":pred_file}]

    out_res = evaluate_multi_files(split=split, input_pred_files=pred_files, task="action_item_detection")
    print(out_res)
