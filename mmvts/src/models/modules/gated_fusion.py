import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/johnarevalo/gmu-mmimdb/blob/master/model.py#L51
class GatedFusionFeature(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # 定义两个线性层来生成文本和视觉特征的门控权重
        self.w_t = nn.Linear(in_size, out_size)  # vis
        self.w_v = nn.Linear(in_size, out_size)  # text
        self.w_z = nn.Linear(in_size * 2, out_size)  # gate

    def forward(self, text_features, vis_features):
        text = torch.tanh(self.w_t(text_features))
        vis = torch.tanh(self.w_v(vis_features))
        mm = torch.cat([text, vis], dim=-1)
        z = torch.sigmoid(self.w_z(mm))
        h = z * text + (1 - z) * vis
        return h


class GatedFusionFeatureV2(nn.Module):
    def __init__(self, in_size, out_size=1):
        super().__init__()
        # 定义两个线性层来生成文本和视觉特征的门控权重
        self.text_gate_layer = nn.Linear(in_size, out_size)
        self.visual_gate_layer = nn.Linear(in_size, out_size)

    def forward(self, text_features, visual_features):
        # 计算每个特征对应的门控权重
        text_gate_scores = self.text_gate_layer(text_features) # [seq_length, 1]
        visual_gate_scores = self.visual_gate_layer(visual_features) # [seq_length, 1]

        # 应用sigmoid激活函数，将分数限制在0和1之间
        text_gate_weights = torch.sigmoid(text_gate_scores)
        visual_gate_weights = torch.sigmoid(visual_gate_scores)

        sums = text_gate_weights + visual_gate_weights
        t_normalized_weights = text_gate_weights / sums
        v_normalized_weights = visual_gate_weights / sums

        # 获得融合特征
        fused_features = t_normalized_weights * text_features + v_normalized_weights * visual_features

        return fused_features


class GatedFusionLogits(nn.Module):
    def __init__(self, in_size, alpha_dim=1):
        super().__init__()
        # 定义两个线性层来生成文本和视觉特征的门控权重
        self.w_t = nn.Linear(in_size, in_size)  # vis
        self.w_v = nn.Linear(in_size, in_size)  # text
        self.w_z = nn.Linear(in_size * 2, alpha_dim)  # gate

    def forward(self, text_logits, vis_logits, text_features, vis_features):
        text = torch.tanh(self.w_t(text_features))
        vis = torch.tanh(self.w_v(vis_features))
        mm = torch.cat([text, vis], dim=-1)
        text_weights = torch.sigmoid(self.w_z(mm))
        h = text_weights * text_logits + (1 - text_weights) * vis_logits
        return h, text_weights


class GlobalFusionLogits(nn.Module):
    def __init__(self, alpha_dim=1):
        super().__init__()
        assert alpha_dim in [1, 2]
        if alpha_dim == 1:
            self.text_weights = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        else:
            self.text_weights = nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
    
    def forward(self, text_logits, vis_logits, text_features, vis_features):
        return self.text_weights * text_logits + (1 - self.text_weights) * vis_logits, self.text_weights
