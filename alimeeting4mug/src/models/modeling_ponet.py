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
#
import shutil

import torch
import random

from torch import nn
from torch.nn import CrossEntropyLoss

from modelscope.models.builder import MODELS
from modelscope.metainfo import Models
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from modelscope.models.nlp.ponet import PoNetPreTrainedModel, PoNetModel
import os
from modelscope.utils.constant import ConfigFields, ModelFile

@MODELS.register_module(
    "token-classification-task", module_name=Models.ponet)
class PoNetForTokenClassification(PoNetPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ponet = PoNetModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            segment_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
        1]``.
    """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ponet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, output_dir, state_dict=None):
        output_file = os.path.join(output_dir, WEIGHTS_NAME)
        # print("save state_dict to %s" % output_file)
        # print("state_dict is ", state_dict)
        torch.save(state_dict, output_file)
        if os.path.isfile(os.path.join(self.model_dir, CONFIG_NAME)):
            self.config.to_json_file(os.path.join(output_dir, CONFIG_NAME))
        if os.path.isfile(os.path.join(self.model_dir, ModelFile.CONFIGURATION)):
            shutil.copy(os.path.join(self.model_dir, ModelFile.CONFIGURATION), os.path.join(output_dir, ModelFile.CONFIGURATION))
