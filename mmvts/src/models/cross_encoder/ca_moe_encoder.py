
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer

from .bert_model import BertCrossLayer
from .moe import MoE

torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(precision=4)


class CoAttentionMoEEncoder(nn.Module):
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

        self.config = config
        bert_config = self.get_config(config)
        self.cross_modal_visual_layers = nn.ModuleList([BertCrossLayer(bert_config, ce_kv_hidden_size=config.ce_kv_hidden_size) for _ in range(config.num_cross_encoder_layers)])
        self.cross_modal_visual_layers.apply(init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config, ce_kv_hidden_size=config.ce_kv_hidden_size) for _ in range(config.num_cross_encoder_layers)])
        self.cross_modal_text_layers.apply(init_weights)
        self.cross_modal_audio_layers = nn.ModuleList([BertCrossLayer(bert_config, ce_kv_hidden_size=config.ce_kv_hidden_size) for _ in range(config.num_cross_encoder_layers)])
        self.cross_modal_audio_layers.apply(init_weights)

        self.moe_layers = nn.ModuleList([MoE(config) for _ in range(config.num_cross_encoder_layers)])
    
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

    def forward(self, attention_mask, text_feat, visual_feat, audio_feat=None):
        if text_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(text_feat.device)
        elif visual_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(visual_feat.device)
        elif audio_feat is not None:
            total_moe_loss = torch.tensor(0.0).to(audio_feat.device)

        # print("attention_mask: ", attention_mask)
        extended_attention_mask = attention_mask[:, None, None, :].to(torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0

        t, v, a = text_feat, visual_feat, audio_feat

        for layer_index, (text_layer, visual_layer, audio_layer) in enumerate(zip(self.cross_modal_text_layers, self.cross_modal_visual_layers, self.cross_modal_audio_layers)):
            
            if t is None:
                v1 = visual_layer(v, a, extended_attention_mask, extended_attention_mask, do_ffn=False)
                a1 = audio_layer(a, v, extended_attention_mask, extended_attention_mask, do_ffn=False)
                v, a = v1[0], a1[0]
            elif v is None:
                t1 = text_layer(t, a, extended_attention_mask, extended_attention_mask, do_ffn=False)
                a1 = audio_layer(a, t, extended_attention_mask, extended_attention_mask, do_ffn=False)
                t, a = t1[0], a1[0]
            elif a is None:
                t1 = text_layer(t, v, extended_attention_mask, extended_attention_mask, do_ffn=False)
                v1 = visual_layer(v, t, extended_attention_mask, extended_attention_mask, do_ffn=False)
                t, v = t1[0], v1[0]
            else:
                av = torch.cat((a, v), dim=-1)
                at = torch.cat((a, t), dim=-1)
                tv = torch.cat((t, v), dim=-1)

                t1 = text_layer(t, av, extended_attention_mask, extended_attention_mask, do_ffn=False)
                v1 = visual_layer(v, at, extended_attention_mask, extended_attention_mask, do_ffn=False)
                a1 = audio_layer(a, tv, extended_attention_mask, extended_attention_mask, do_ffn=False)
                t, v, a = t1[0], v1[0], a1[0]
            
            # moe
            if t is None:
                z = torch.cat((v, a), dim=1)    # cat in seq dim
            elif v is None:
                z = torch.cat((t, a), dim=1) 
            elif a is None:
                z = torch.cat((t, v), dim=1) 
            else:
                z = torch.cat((t, v, a), dim=1) 

            batch_size, seq_length, hidden_size = z.shape
            z = z.view(batch_size * seq_length, hidden_size)

            if self.config.cross_moe_share_in_layers:
                layer_output, moe_loss = self.moe(z)
            else:
                layer_output, moe_loss = self.moe_layers[layer_index](z)
            
            layer_output = layer_output.view(batch_size, seq_length, hidden_size)

            if t is None:
                v, a = torch.chunk(layer_output, 2, dim=1)
            elif v is None:
                t, a = torch.chunk(layer_output, 2, dim=1)
            elif a is None:
                t, v = torch.chunk(layer_output, 2, dim=1)
            else:
                t, v, a = torch.chunk(layer_output, 3, dim=1)

            total_moe_loss += moe_loss

        text_feat, visual_feat, audio_feat = t, v, a
        return text_feat, visual_feat, audio_feat, total_moe_loss
