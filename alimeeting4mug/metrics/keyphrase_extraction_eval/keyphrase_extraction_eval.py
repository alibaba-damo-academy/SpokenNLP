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
# limitations under the License.from rouge import Rouge
""" kpe metric. """

import importlib
import numpy as np
from typing import List, Optional, Union
import datasets
from rouge import Rouge

_CITATION = """"""

_DESCRIPTION = """"""

_KWARGS_DESCRIPTION = """"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class KpeEval(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[""],
        )

    def _compute(
            self,
            predictions,
            references,
    ):
        scores = {}
        score_sum = 0.0
        for num in [10, 15, 20]:
            predictions_at_num = [pred[:num] for pred in predictions]
            approximate_match_score = self.calculateCorpusApproximateMatchScore(predictions_at_num, references)
            rouge_score = self.calculateRouge(predictions_at_num, references)

            for k, v in approximate_match_score.items():
                scores[k + "@%d" % num] = v
                score_sum += v
            for k, v in rouge_score.items():
                scores[k + "@%d" % num] = v
                score_sum += v

        scores["score"] = score_sum / len(scores.keys())
        return scores

    def calculateCorpusApproximateMatchScore(self, keywords, goldenwords):
        # print("calculateCorpusApproximateMatchScore...")
        partial_f1_list = []
        for example_keywords, example_goldenwords in zip(keywords, goldenwords):
            example_score = self.calculateExampleApproximateMatchScore(example_keywords, example_goldenwords)
            partial_f1_list.append(example_score["partial_f1"])

        partial_f1 = sum(partial_f1_list) * 1.0 / len(partial_f1_list)
        # print("partial_f1: ", partial_f1)
        return {"partial_f1": partial_f1}


    def calculateExampleApproximateMatchScore(self, keywords, goldenwords):

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

        # keywords = sum(keywords, [])
        # goldenwords = sum(goldenwords, [])
        # print("keywords: ", keywords)
        # print("goldenwords: ", goldenwords)

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

    def calculateRouge(self, keywords, goldenwords):
        keywords = [" ".join(_) for _ in keywords]
        goldenwords = [" ".join(_) for _ in goldenwords]
        # print("keywords: ", keywords)
        # print("goldenwords: ", goldenwords)
        rouge = Rouge()
        scores = rouge.get_scores(hyps=keywords, refs=goldenwords, avg=True)
        return {
            'exact_f1': scores['rouge-1']['f']
        }
