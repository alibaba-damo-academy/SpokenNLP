
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO trainable audio encoder
        self.config = config    

    def forward(
        self, 
        example_ids, 
        audio_embeds,
    ):
        audio_feature = audio_embeds
        return audio_feature
