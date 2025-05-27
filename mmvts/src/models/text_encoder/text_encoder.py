
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = self.get_encoder(config)
        
        # self.text_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # self.text_norm = nn.LayerNorm(config.hidden_size)
        # self.text_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def get_encoder(self, config):
        if "bert" in config.text_encoder_name_or_path:
            self.encoder_type = "bert"

            from transformers.models.bert.modeling_bert import BertModel

            if config.init_model:
                text_encoder = BertModel.from_pretrained(
                    config.text_encoder_name_or_path,
                    config = config,
                    cache_dir = config.cache_dir,
                    revision = config.model_revision,
                )
            else:
                text_encoder = BertModel(config)

        else:
            self.encoder_type = "lf"
            from transformers.models.longformer.modeling_longformer import LongformerModel
            
            if config.init_model:
                text_encoder = LongformerModel.from_pretrained(
                    config.text_encoder_name_or_path,
                    config = config,
                    cache_dir = config.cache_dir,
                    revision = config.model_revision,
                )
            else:
                text_encoder = LongformerModel(config)

        return text_encoder

    def forward(self,
        input_ids,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
        if self.encoder_type == "bert":
            text_outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask if head_mask is not None else None,
                token_type_ids=token_type_ids,
                position_ids=position_ids if position_ids is not None else None,
                inputs_embeds=inputs_embeds if inputs_embeds is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            text_outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask if head_mask is not None else None,
                token_type_ids=token_type_ids,
                position_ids=position_ids if position_ids is not None else None,
                inputs_embeds=inputs_embeds if inputs_embeds is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        text_feature = text_outputs[0]
        # text_feature = self.text_proj(text_feature)
        # text_feature = self.text_norm(text_feature)
        # text_feature = self.text_dropout(text_feature)
        return text_feature
        