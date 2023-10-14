

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SentenceFeaturesExtractor


class TSSP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_tssp_labels)
    
    def forward(
        self, 
        sent_token_mask,
        da_seq_output,
        da_sent_pair_orders,
        ):
        tssp_loss = torch.tensor(0.0).to(da_seq_output.device)
        if self.config.tssp_loss_weight == 0:
            return loss
        
        sent_extractor = SentenceFeaturesExtractor()
        sent_features, _ = sent_extractor(
            sequence_output=da_seq_output,
            sent_token_mask=sent_token_mask,
        )
        logits = self.classifier(sent_features)
        tssp_labels = da_sent_pair_orders[da_sent_pair_orders != -100]
        loss_fct = torch.nn.CrossEntropyLoss()
        tssp_loss = loss_fct(logits.reshape(-1, self.config.num_tssp_labels), tssp_labels.reshape(-1))

        return self.config.tssp_loss_weight * tssp_loss
