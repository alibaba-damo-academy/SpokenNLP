
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerProjector(nn.Module):
    def __init__(self, config):
        super(TransformerProjector, self).__init__()
        self.vis_proj = nn.Linear(config.hidden_size_vis, config.hidden_size)
        self.vis_norm = nn.LayerNorm(config.hidden_size)
        self.vis_dropout = nn.Dropout(p=0.1)

        self.shared = getattr(config, "proj_shared", False)
        self.proj_skip = getattr(config, "proj_skip", False)
        input_size = config.hidden_size
        hidden_size = config.hidden_size
        dropout_prob = config.hidden_dropout_prob
        num_layers = getattr(config, "proj_num_layers", 1)
        nhead = getattr(config, "proj_nhead", 12)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size)
        self.proj = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        if self.shared:
            self.proj_vis = self.proj
            self.layernorm_vis = self.layernorm
            self.dropout_vis = self.dropout
        else:
            encoder_layer_vis = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size)
            self.proj_vis = nn.TransformerEncoder(encoder_layer_vis, num_layers=num_layers)
            self.layernorm_vis = nn.LayerNorm(hidden_size)
            self.dropout_vis = nn.Dropout(dropout_prob)
        
    def forward(self, text_feature, vis_feature=None):
        # add batch dim
        text_feature = text_feature.unsqueeze(0)
        if vis_feature is not None:
            vis_feature = self.vis_dropout(self.vis_norm(self.vis_proj(vis_feature)))
            vis_feature = vis_feature.unsqueeze(0)

        if self.proj_skip:
            text_residual = text_feature
            text_feature = self.layernorm(self.dropout(self.proj(text_feature)) + text_residual)
        else:
            text_feature = self.layernorm(self.dropout(self.proj(text_feature)))

        if vis_feature is not None:
            if self.proj_skip:
                vis_residual = vis_feature
                vis_feature = self.layernorm_vis(self.dropout_vis(self.proj_vis(vis_feature)) + vis_residual)
            else:
                vis_feature = self.layernorm_vis(self.dropout_vis(self.proj_vis(vis_feature)))
        
        # remove batch dim
        text_feature = text_feature.squeeze(0)
        vis_feature = vis_feature.squeeze(0)

        return text_feature, vis_feature