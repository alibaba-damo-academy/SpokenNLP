# Copyright 2020 The HuggingFace Datasets Authors.
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
""" seqeval metric. """

import importlib
import numpy as np
from typing import List, Optional, Union

import scipy
from seqeval.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from segeval.window.pk import pk as PK
from segeval.window.windowdiff import window_diff as WD

import datasets


_CITATION = """\
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
"""

_DESCRIPTION = """\
seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script conlleval, which can be used for
measuring the performance of a system that has processed the CoNLL-2000 shared task data.

seqeval supports following formats:
IOB1
IOB2
IOE1
IOE2
IOBES

See the [README.md] file at https://github.com/chakki-works/seqeval for more information.
"""

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.

Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    suffix: True if the IOB prefix is after type, False otherwise. default: False
    scheme: Specify target tagging scheme. Should be one of ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"].
        default: None
    mode: Whether to count correct entity labels with incorrect I/B tags as true positives or not.
        If you want to only count exact matches, pass mode="strict". default: None.
    sample_weight: Array-like of shape (n_samples,), weights for individual samples. default: None
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be on of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.

Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:

    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = datasets.load_metric("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Seqeval(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/chakki-works/seqeval",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/chakki-works/seqeval"],
            reference_urls=["https://github.com/chakki-works/seqeval"],
        )

    def _compute(
        self,
        predictions,
        references,
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = "warn",
    ):
        new_predictions, new_references= [], []
        actual_boundary_num, pred_boundary_num = 0, 0
        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                continue
            new_predictions.append(pred)
            new_references.append(ref)

            actual_boundary_num += len([v for v in ref if v == "B-EOP"])
            pred_boundary_num += len([v for v in pred if v == "B-EOP"])

        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")
        report = classification_report(
            y_true=new_references,
            y_pred=new_predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        scores = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": int(score["support"]),
            }
            for type_name, score in report.items()
        }
        scores["overall_precision"] = overall_score["precision"]
        scores["overall_recall"] = overall_score["recall"]
        scores["overall_f1"] = overall_score["f1-score"]
        scores["overall_accuracy"] = accuracy_score(y_true=new_references, y_pred=new_predictions)
        
        scores["actual_boundary_num"] = actual_boundary_num
        scores["pred_boundary_num"] = pred_boundary_num
        print("scores: ", scores)
        
        return scores

    def compute_window_metric(self, predictions, references, prefix=""):
        # predictions and references are list of list.
        # each list is segmentation of each example.
        # each list consists of some 0 and 1, where 1 means end sentence of topic.

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
        
        n = len(predictions)
        case_results = {
            "1-pk": [],
            "1-wd": [],
        }
        for i, (y_pred_label, y_true_label) in enumerate(zip(predictions, references)):
            try:
                y_pred_mass = mass_from_end_label_sequence(y_pred_label)
                y_true_mass = mass_from_end_label_sequence(y_true_label)
                assert sum(y_pred_mass) == sum(y_true_mass)
                pk, wd = compute_metric_from_mass_data(y_pred_mass, y_true_mass)
                case_results["1-pk"].append(1 - pk)
                case_results["1-wd"].append(1 - wd)
            except Exception as e:
                pass
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
            prefix + "pk": 1 - total_result["1-pk"],
            prefix + "wd": 1 - total_result["1-wd"],
            prefix + "avg_pred_cnt": avg_pred_cnt,
            prefix + "avg_true_cnt": avg_true_cnt,
        }
    
    def compute_accuracy(self, predictions, labels):
        # input is 2d array
        total, right = 0, 0
        for prediciton, label in zip(predictions, labels):
            for p, l in zip(prediciton, label):
                total += 1
                right += p == l
        return right * 1.0 / total

    def compute_metric_example_level(self, predictions_logits, labels, label_list, custom_args, data_args, reverse_logits=False, mode="test"):
        predictions = [np.argmax(np.array(logits), axis=-1) for logits in predictions_logits]
        seg_point_predictions_scores = [scipy.special.softmax(np.array(logits), axis=-1)[:, 0] for logits in predictions_logits]
        
        if reverse_logits:
            predictions = [[1 - v for v in pred] for pred in predictions]
        
        # label_list is ['B-EOP', 'O'] so label_list[0] is seg point which means end sentence of topic
        for label in labels:
            for l in label:
                assert l != -100, "l == -100"    
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.compute(predictions=true_predictions, references=true_labels)

        # accuracy of 01
        results["accuracy"] = self.compute_accuracy(true_predictions, true_labels)

        true_labels_binary = [
            [int(not l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]       # convert label from label_list to another representation. 1 means seg which is end sentence of topic, 0 means not seg
        custom_eval_results = {}

        if custom_args.threshold is not None:
            # 1 means seg which is end sentence of topic
            true_predictions_binary = [
                [1 if prob >= custom_args.threshold else 0 for (prob, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(seg_point_predictions_scores, labels)
            ]
            
            results_by_threshold = self.compute_window_metric(predictions=true_predictions_binary, references=true_labels_binary, prefix="threshold_%s_example_level_" % str(custom_args.threshold))
            custom_eval_results.update(results_by_threshold)
        
        # if custom_args.topk is not None:
        #     prefix = "topk_%s_example_level_" % str(custom_args.topk)
        #     para_level_predictions_scores = [ 
        #         [prob for prob, l in zip(prediction, label) if l != -100]
        #         for prediction, label in zip(seg_point_predictions_scores, labels)
        #     ]
        #     sorted_scores_tuple = [sorted([(v, i) for i, v in enumerate(prediction)], reverse=True) for prediction in para_level_predictions_scores]
        
        #     # choose topk scores
        #     keep_indices = [np.array([scores_tuple[i][1] for i in range(min(len(scores_tuple), custom_args.topk))]) for scores_tuple in sorted_scores_tuple]
        #     kth_scores = [scores_tuple[min(len(scores_tuple), custom_args.topk)][0] for scores_tuple in sorted_scores_tuple]
        #     kth_scores_avg = round(sum(kth_scores) / len(kth_scores), 3)
            
        #     true_predictions_binary = [np.array([0] * len(prediction)) for prediction in para_level_predictions_scores]
        #     for true_prediction, keep_index in zip(true_predictions_binary, keep_indices):
        #         if len(keep_index) != 0:
        #             true_prediction[keep_index] = 1            
        #     true_predictions_binary = [true_prediction.tolist() for true_prediction in true_predictions_binary]

        #     results_by_topk = self.compute_window_metric(predictions=true_predictions_binary, references=true_labels_binary, prefix=prefix)
        #     results_by_topk[prefix + "kth_scores_avg"] = kth_scores_avg
        #     custom_eval_results.update(results_by_topk)

        #     # choose topk scores which are bigger than threshold
        #     if custom_args.topk_with_threshold:
        #         assert custom_args.threshold != None
        #         keep_indices = [np.array([scores_tuple[i][1] for i in range(min(len(scores_tuple), custom_args.topk)) if scores_tuple[i][0] >= custom_args.threshold]) for scores_tuple in sorted_scores_tuple]
        #         true_predictions_binary = [np.array([0] * len(prediction)) for prediction in para_level_predictions_scores]
        #         for true_prediction, keep_index in zip(true_predictions_binary, keep_indices):
        #             if len(keep_index) != 0:
        #                 true_prediction[keep_index] = 1            
        #         true_predictions_binary = [true_prediction.tolist() for true_prediction in true_predictions_binary]

        #         results_by_topk_with_threshold = self.compute_window_metric(predictions=true_predictions_binary, references=true_labels_binary, prefix="topk_%s_with_threshold_%s_example_level_" % (str(custom_args.topk), str(custom_args.threshold)))
        #         custom_eval_results.update(results_by_topk_with_threshold)

        # if custom_args.f1_at_k is not None and custom_args.f1_at_k != 0:
        #     true_predictions_binary = [
        #         [1 if prob >= custom_args.threshold else 0 for (prob, l) in zip(prediction, label) if l != -100]
        #         for prediction, label in zip(seg_point_predictions_scores, labels)
        #     ]
        #     soft_predictions_binary = []
        #     for predcition, label in zip(true_predictions_binary, true_labels_binary):
        #         for i, p in enumerate(predcition):
        #             if p == 0 or (p == 1 and label[i] == 1):
        #                 continue
        #             left = max(0, i - custom_args.f1_at_k)
        #             right = min(len(predcition) - 1, i + custom_args.f1_at_k)
        #             for j in range(left, right + 1):
        #                 if label[j] == 1:
        #                     predcition[i] = 0
        #                     predcition[j] = 1
        #                     break
        #         soft_predictions_binary.append(predcition)
        #     results_by_f1_at_k = self.compute_window_metric(predictions=soft_predictions_binary, references=true_labels_binary, prefix="f1@%d_example_level_" % custom_args.f1_at_k)
        #     custom_eval_results.update(results_by_f1_at_k)
        final_results = {}
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
        else:
            final_results.update(
                {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
            }
            )
        
        final_results.update(custom_eval_results)
        return final_results
