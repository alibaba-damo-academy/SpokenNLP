
import torch.nn as nn


class VisOcrEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, current_mode):
        pass