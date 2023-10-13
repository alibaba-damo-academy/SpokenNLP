
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cssl import CSSL
from .tssp import TSSP
from .utils import get_loss_fct, EopPairCosineSimilarity


class LossCalculator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.eop_pair_cos_sim = EopPairCosineSimilarity(temp=self.config.ts_score_predictor_cos_temp)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cssl = CSSL(config=config)
        self.tssp = TSSP(config=config)

    def multiple2one_pooling(self, src, index, dim=1, pool_type="amax"):
        # shape of result is the same as src by padding empty tensor
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)
    
    def forward(
        self, 
        sequence_output, 
        labels,
        extract_eop_segment_ids=None,
        eop_index_for_aggregate_batch_eop_features=None,
        sent_token_mask=None,
        sent_pair_orders=None,
        da_example_flag=False,
        ):
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(sequence_output.device)
        
        # compute cosine similarity of eop pairs consits of eop and next eop
        batch_eop_pair_cos_sim, batch_eop_labels = self.eop_pair_cos_sim(sequence_output, labels)
        
        # compute ts loss
        if self.config.ts_score_predictor == "lt":
            logits = self.classifier(sequence_output)
            loss_fct = get_loss_fct(gamma=self.config.focal_loss_gamma, weight_label_zero=self.config.weight_label_zero, device=sequence_output.device)
            ts_loss = loss_fct(logits.reshape(-1, self.config.num_labels), labels.reshape(-1))
        elif self.config.ts_score_predictor == "cos":
            loss_fct = torch.nn.BCEWithLogitsLoss()
            ts_loss = loss_fct(batch_eop_pair_cos_sim.reshape(-1), batch_eop_labels.reshape(-1).float())
            logits = torch.sigmoid(batch_eop_pair_cos_sim)
        else:
            raise ValueError("not supported ts_score_predictor %s" % ts_score_predictor)
        loss += self.config.ts_loss_weight * ts_loss

        # compute cssl loss
        if da_example_flag is False and self.config.cl_loss_weight != 0:
            assert extract_eop_segment_ids is not None and eop_index_for_aggregate_batch_eop_features is not None
            cl_loss = self.cssl(
                sequence_output=sequence_output,
                labels=labels,
                extract_eop_segment_ids=extract_eop_segment_ids,
                eop_index_for_aggregate_batch_eop_features=eop_index_for_aggregate_batch_eop_features,
            )
            loss += self.config.cl_loss_weight * cl_loss
        
        # compute tssp loss
        if da_example_flag is True and self.config.tssp_loss_weight != 0:
            tssp_loss = self.tssp(
                sent_token_mask=sent_token_mask,
                da_seq_output=sequence_output,
                da_sent_pair_orders=sent_pair_orders,
            )
            loss += self.config.tssp_loss_weight * tssp_loss
        
        return loss, logits, batch_eop_pair_cos_sim
