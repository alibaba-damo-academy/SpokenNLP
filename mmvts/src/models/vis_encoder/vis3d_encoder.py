
import torch.nn as nn


class Vis3dEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, current_mode):
        pass