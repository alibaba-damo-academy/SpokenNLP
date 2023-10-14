
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .modules.loss_calculator import LossCalculator


class BertWithDAForSentenceLabelingTopicSegmentation(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.loss_calculator = LossCalculator(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict = False,
        sent_level_labels=None,
        extract_eop_segment_ids=None,
        eop_index_for_aggregate_batch_eop_features=None,
        sent_pair_orders=None,
        sent_token_mask=None,
    ):
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(input_ids.device)

        anchor_outputs = self.bert(
            input_ids[:, 0],
            attention_mask=attention_mask[:, 0],
            head_mask=head_mask[:, 0] if head_mask is not None else None,
            token_type_ids=token_type_ids[:, 0],
            position_ids=position_ids[:, 0] if position_ids is not None else None,
            inputs_embeds=inputs_embeds[:, 0] if inputs_embeds is not None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        anchor_seq_output = anchor_outputs[0]
        anchor_seq_output = self.dropout(anchor_seq_output)
        
        if self.config.do_da_ts or self.config.do_tssp:
            da_outputs = self.bert(
                input_ids[:, 1],
                attention_mask=attention_mask[:, 1],
                head_mask=head_mask[:, 1] if head_mask is not None else None,
                token_type_ids=token_type_ids[:, 1],
                position_ids=position_ids[:, 1] if position_ids is not None else None,
                inputs_embeds=inputs_embeds[:, 1] if inputs_embeds is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            da_seq_output = da_outputs[0]
            da_seq_output = self.dropout(da_seq_output)

        anchor_labels, da_labels = labels[:, 0], labels[:, 1]

        logits, batch_eop_pair_cos_sim = None, None

        if labels is not None:
            anchor_loss, anchor_logits, batch_eop_pair_cos_sim = self.loss_calculator(
                    sequence_output=anchor_seq_output,
                    labels=anchor_labels,
                    extract_eop_segment_ids=extract_eop_segment_ids[:, 0],
                    eop_index_for_aggregate_batch_eop_features=eop_index_for_aggregate_batch_eop_features[:, 0],
                )
            loss += anchor_loss
            logits = torch.cat((anchor_logits.unsqueeze(1), anchor_logits.unsqueeze(1)), dim=1)
            
            if self.config.do_da_ts or self.config.do_tssp:
                da_loss, da_logits, _ = self.loss_calculator(
                        sequence_output=da_seq_output,
                        labels=da_labels,
                        extract_eop_segment_ids=extract_eop_segment_ids[:, 1],
                        eop_index_for_aggregate_batch_eop_features=eop_index_for_aggregate_batch_eop_features[:, 1],
                        sent_token_mask=sent_token_mask[:, 1],
                        sent_pair_orders=sent_pair_orders[:, 1],
                        da_example_flag=True,
                    )
                loss += da_loss
                logits = torch.cat((anchor_logits.unsqueeze(1), da_logits.unsqueeze(1)), dim=1)
            
        output = (logits, batch_eop_pair_cos_sim,)
        output += anchor_outputs[2:]
        return ((loss,) + output) if loss is not None else output
