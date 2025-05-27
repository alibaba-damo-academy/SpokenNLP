
import torch.nn as nn

from .linear_predictor import LinearPredictor


class TopicSegPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictor = LinearPredictor(config)
    
    def forward(
        self,
        labels,
        text_features=None,
        visual_features=None,
        audio_features=None,
        moe_loss=None,
        projected_text_features=None,
        projected_visual_features=None,
        projected_audio_features=None,
    ):
        logits, loss = self.predictor(labels, text_features, visual_features, audio_features, moe_loss, projected_text_features, projected_visual_features, projected_audio_features)
        return logits, loss
