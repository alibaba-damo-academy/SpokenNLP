
import torch
import torch.nn as nn

from .base_predictor import BasePredictor


class LinearPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.classifier = nn.Linear(config.in_predictor_hidden_size, 2)

    def get_logits(self, fused_features):
        logits = self.classifier(fused_features)
        return logits

    def compute_modal_logits(self, fused_features):
        # 适用于 cat 的操作
        assert self.config.fuse_type == "cat"
        unimodal_hidden_size = self.config.in_predictor_hidden_size // 3
        text_features, vis_features, audio_features = torch.chunk(fused_features, 3, dim=1)
        # 获得原始层的参数
        original_weights = self.classifier.weight.data
        original_bias = self.classifier.bias.data

        # 将权重和偏差拆分为各部分，每部分处理对应的输入特征
        weights1, weights2, weights3 = original_weights[:,:unimodal_hidden_size], original_weights[:,unimodal_hidden_size:unimodal_hidden_size*2], original_weights[:,unimodal_hidden_size*2:], 
        bias1, bias2, bias3 = original_bias / 3, original_bias / 3, original_bias / 3

        # 对输入x的每个部分应用相应的权重和偏差
        output1 = torch.matmul(text_features, weights1.t()) + bias1
        output2 = torch.matmul(vis_features, weights2.t()) + bias2
        output3 = torch.matmul(audio_features, weights3.t()) + bias3
        return output1, output2, output3

    def forward(self, labels, text_features=None, visual_features=None, audio_features=None, moe_loss=None, projected_text_features=None, projected_visual_features=None, projected_audio_features=None):
        # labels有batch维度，其余没有batch维度
        fused_features = self.fuse_features(text_features, visual_features, audio_features)
        logits = self.get_logits(fused_features)

        if self.config.fuse_type == "cat":
            text_logits, vis_logits, audio_logits = self.compute_modal_logits(fused_features)
        else:
            text_logits, vis_logits, audio_logits = None, None, None

        loss_dict = self.loss_layer(
            labels=labels, 
            logits=logits,
            text_features=text_features,
            visual_features=visual_features,
            audio_features=audio_features,
            fused_features=fused_features,
            moe_loss=moe_loss,
            projected_text_features=projected_text_features,
            projected_visual_features=projected_visual_features,
            projected_audio_features=projected_audio_features,
        )

        if self.config.out_modal_prob:
            total_logits = torch.cat((logits, text_logits, vis_logits), dim=1)  # seq * 6
            return total_logits, loss_dict["loss"]["total_loss"]
        else:
            return logits, loss_dict["loss"]["total_loss"]