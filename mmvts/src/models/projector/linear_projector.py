
import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, config):
        super(LinearProjector, self).__init__()

        self.proj_text = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm_text = nn.LayerNorm(config.hidden_size)
        self.dropout_text = nn.Dropout(config.hidden_dropout_prob)

        self.proj_vis = nn.Linear(config.hidden_size_vis, config.hidden_size)
        self.layernorm_vis = nn.LayerNorm(config.hidden_size)
        self.dropout_vis = nn.Dropout(config.hidden_dropout_prob)

        self.proj_audio = nn.Linear(config.hidden_size_audio, config.hidden_size)
        self.layernorm_audio = nn.LayerNorm(config.hidden_size)
        self.dropout_audio = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, text_feature=None, vis_feature=None, audio_feature=None):
        if text_feature is not None:
            text_feature = self.dropout_text(self.layernorm_text(self.proj_text(text_feature)))

        if vis_feature is not None:
            vis_feature = self.dropout_vis(self.layernorm_vis(self.proj_vis(vis_feature)))
        
        if audio_feature is not None:
            audio_feature = self.dropout_audio(self.layernorm_audio(self.proj_audio(audio_feature)))
        
        return text_feature, vis_feature, audio_feature