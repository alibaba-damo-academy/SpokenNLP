
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer

from .bert_model import BertMoELayer


class MergeAttentionMoEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        bert_config = self.get_config(config)
        self.cross_modal_layers = nn.ModuleList([BertMoELayer(bert_config, config) for _ in range(config.num_cross_encoder_layers)])
        self.cross_modal_layers.apply(init_weights)
    
    def get_config(self, config):
        bert_config = BertConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_cross_encoder_layers,
            num_attention_heads=config.num_cross_encoder_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_seq_length,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )
        return bert_config

    def forward(self, attention_mask, text_feat=None, visual_feat=None, audio_feat=None):
        if text_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(text_feat.device)
        elif visual_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(visual_feat.device)
        elif audio_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(audio_feat.device)

        if text_feat is None:
            z = torch.cat((visual_feat, audio_feat), dim=1) 
            cat_mask = torch.cat((attention_mask, attention_mask), dim=1)
        elif visual_feat is None:
            z = torch.cat((text_feat, audio_feat), dim=1) 
            cat_mask = torch.cat((attention_mask, attention_mask), dim=1)
        elif audio_feat is None:
            z = torch.cat((text_feat, visual_feat), dim=1) 
            cat_mask = torch.cat((attention_mask, attention_mask), dim=1)
        else:
            assert text_feat is not None and visual_feat is not None and audio_feat is not None
            z = torch.cat((text_feat, visual_feat, audio_feat), dim=1) 
            cat_mask = torch.cat((attention_mask, attention_mask, attention_mask), dim=1)

        extended_attention_mask = cat_mask[:, None, None, :].to(torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0

        # print("z.shape: ", z.shape)
        for cross_modal_layer in self.cross_modal_layers:
            z, moe_loss = cross_modal_layer(z, extended_attention_mask)
            total_moe_loss += moe_loss
        
        if text_feat is None:
            visual_feat, audio_feat = torch.chunk(z, 2, dim=1)
        elif visual_feat is None:
            text_feat, audio_feat = torch.chunk(z, 2, dim=1)
        elif audio_feat is None:
            text_feat, visual_feat = torch.chunk(z, 2, dim=1)
        else:
            text_feat, visual_feat, audio_feat = torch.chunk(z, 3, dim=1)

        return text_feat, visual_feat, audio_feat, total_moe_loss
