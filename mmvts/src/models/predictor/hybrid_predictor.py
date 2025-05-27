
import torch
import torch.nn as nn

from .base_predictor import BasePredictor
from ..modules.gated_fusion import *


class HybridPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 学习分类概率
        self.vis_classifier = nn.Linear(config.hidden_size, 2)
        self.text_classifier = nn.Linear(config.hidden_size, 2)
        self.mm_classifier = nn.Linear(config.in_predictor_hidden_size, 2)

        # 学习如何组合分类概率
        self.w_t = nn.Linear(config.hidden_size, config.hidden_size)  # vis
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)  # text
        self.w_mm = nn.Linear(config.hidden_size * 2, 3)               # gate

        self.mm_weight = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.text_weight = nn.Parameter(torch.tensor(0.3, requires_grad=True))
        self.vis_weight = nn.Parameter(torch.tensor(0.2, requires_grad=True))

    
    def get_weights(self, text_features, visual_features):
        if self.config.predictor_hybrid_weight_type == "p":         # parameter
            weights = torch.cat((self.mm_weight.unsqueeze(0), self.text_weight.unsqueeze(0), self.vis_weight.unsqueeze(0)))
            weights = torch.softmax(weights, dim=-1)
            weights = weights.repeat(text_features.shape[0], 1)
        elif self.config.predictor_hybrid_weight_type == "l":   # linear
            text = torch.tanh(self.w_t(text_features))
            vis = torch.tanh(self.w_v(visual_features))
            mm = torch.cat([text, vis], dim=-1)
            weights = torch.softmax(self.w_mm(mm), dim=-1)
        return weights
        
    def get_logits(self, fused_features, text_features, visual_features):
        mm_logits = self.mm_classifier(fused_features)
        text_logits = self.text_classifier(text_features)
        vis_logits = self.vis_classifier(visual_features)
        all_logits = torch.cat((mm_logits.unsqueeze(0), text_logits.unsqueeze(0), vis_logits.unsqueeze(0)), dim=0)  # shape is 3 * n * 2

        weights = self.get_weights(text_features, visual_features)    # shape is n * 3
        weights = weights.unsqueeze(2).permute(1, 0, 2).repeat_interleave(2, dim=2)    # shape is 3 * n * 2

        weighted_logits = all_logits * weights
        weighted_logits = weighted_logits.transpose(0, 1)       # shape is n * 3 * 2

        if self.config.predictor_hybrid_pooling == "max":
            final_logits = torch.max(weighted_logits, dim=1).values
        elif self.config.predictor_hybrid_pooling == "mean":
            final_logits = torch.mean(weighted_logits, dim=1)
        else:
            raise ValueError("not supported {}".format(self.config.predictor_hybrid_pooling))

        return final_logits, mm_logits, text_logits, vis_logits, weights

    def forward(self, labels, text_features=None, visual_features=None):
        # labels有batch维度，其余没有batch维度
        fused_features = self.fuse_features(text_features, visual_features)
        final_logits, mm_logits, text_logits, vis_logits, weights = self.get_logits(fused_features, text_features, visual_features)
        
        loss_dict = self.loss_layer(
            labels=labels, 
            logits=final_logits,
            text_features=text_features,
            visual_features=visual_features,
            fused_features=fused_features,
            mm_logits=mm_logits,
            text_logits=text_logits,
            vis_logits=vis_logits,
        )
        
        return final_logits, loss_dict["loss"]["total_loss"]