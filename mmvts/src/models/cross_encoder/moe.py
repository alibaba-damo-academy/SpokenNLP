
import torch.nn as nn


class MoE(nn.Module):
    # TODO: put here implement pytorch version MoE based on TensorFlow implementation:
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
    def __init__(self, config):
        super(MoE, self).__init__()

    def forward(self, x):
        y = x
        loss = 0
        return y, loss
