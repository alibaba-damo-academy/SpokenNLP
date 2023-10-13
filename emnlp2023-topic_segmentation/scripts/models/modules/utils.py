
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class SentenceFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        sequence_output,
        sent_token_mask,
        ):
        # input sequence_output and sent_token_mask, where
        # sequence_output is (bs, seq_length, hidden_size)
        # sent_token_mask is (bs, seq_length), each value is 1, 0 or -100, where 1 and 0 means bos token, and 0 means end sentence of topic
        
        # return sent_features and topic_segment_ids, where
        # sent_features is sentence level features, shape is (number of sent, hidden_size) which has compress the batch vector into one dimension
        # topic_segment_ids is a list like [0,0,0, 1, 2,2,2,2, 3,3,3] where same ids belong to the same topic
        
        sent_features = sequence_output[sent_token_mask != -100]

        sent_labels = [l[l != -100] for l in sent_token_mask]
        topic_segment_ids = []     # for recording topic level segment id. [a1, a2, a3, b1, c1, c2] will have topic_segment_ids [0, 0, 0, 1, 2, 2]
        seg_id = 0
        for example_sent_labels in sent_labels:
            if len(example_sent_labels) == 0:
                continue
            for l in example_sent_labels:
                topic_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_sent_labels[-1] == 1:
                seg_id += 1

        return sent_features, torch.tensor(topic_segment_ids).to(sent_features.device)
    

class EopFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def multiple2one_pooling(self, src, index, dim=1, pool_type="amax"):\
        # shape of result is the same as src by padding empty tensor
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)
 
    def forward(
        self,
        sequence_output,
        labels,
        extract_eop_segment_ids,
        eop_index_for_aggregate_batch_eop_features,
        ):
        # return sent_features and topic_segment_ids, where
        # sent_features is eop sentence level features, shape is (number of sent/eop, hidden_size) which has Compress the batch vector into one dimension
        # topic_segment_ids is a list like [0,0,0, 1, 2,2,2,2, 3,3,3] where same ids belong to the same topic
        
        loss = None
        bs, seq_length, hidden_size = sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]

        # compute contrastive learning loss
        eop_level_output = self.multiple2one_pooling(sequence_output, extract_eop_segment_ids, pool_type="amax")    # contains cls, shape is (bs, seq_len, hidden_size)
        tmp_eop_index = eop_index_for_aggregate_batch_eop_features + torch.arange(bs).to(sequence_output.device).unsqueeze(1).expand_as(eop_index_for_aggregate_batch_eop_features) * seq_length
        tmp_eop_index = tmp_eop_index.reshape(-1)
        eop_index_for_aggregate_batch_eop_features = eop_index_for_aggregate_batch_eop_features.reshape(-1)
        eop_index = tmp_eop_index[eop_index_for_aggregate_batch_eop_features != 0]
        eop_features = eop_level_output.reshape(bs * seq_length, -1)[eop_index]     # shape is (number of eop, hidden_size)

        eop_labels = [l[l != -100] for l in labels]
        topic_segment_ids = []     # for recording topic level segment id. [a1, a2, a3, b1, c1, c2] will have topic_segment_ids [0, 0, 0, 1, 2, 2]
        seg_id = 0
        for example_eop_labels in eop_labels:
            if len(example_eop_labels) == 0:
                continue
            for l in example_eop_labels:
                topic_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_eop_labels[-1] == 1:
                seg_id += 1

        return eop_features, topic_segment_ids


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        if self.temp == 0:
            # compute dot product matrix rather than cosine similarity matrix
            x = x.squeeze(1)
            y = y.squeeze(0)
            # then x and y are same shape with (k, hidden_size)
            return torch.matmul(x, y.t())
        else:
            return self.cos(x, y) / self.temp


class EopPairCosineSimilarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.cos_sim_fct = Similarity(temp=temp)

    def forward(self, sequence_output, labels):
        batch_eop_pair_cos_sim, batch_eop_labels = [], []
        max_eop_sent_cnt = 0
        for example_sequence_output, example_labels in zip(sequence_output, labels):
            eop_mask = example_labels != -100
            batch_eop_labels.append(example_labels[eop_mask])
            example_eop_sent_out = example_sequence_output[eop_mask]
            eop_sent_cnt = example_eop_sent_out.shape[0]
            max_eop_sent_cnt = max(max_eop_sent_cnt, eop_sent_cnt)
            sent_index = torch.arange(0, eop_sent_cnt)
            next_sent_index = (sent_index + 1) % eop_sent_cnt   # last eop should compute cosine similarity with end sentence
            next_sent_out = example_eop_sent_out[next_sent_index]

            cos_sim = self.cos_sim_fct(example_eop_sent_out, next_sent_out)      # cos_sim is like torch.tensor([0.9, 0.7, 0.8, ...])
            batch_eop_pair_cos_sim.append(cos_sim)

        for i, (cos_sim, eop_labels) in enumerate(zip(batch_eop_pair_cos_sim, batch_eop_labels)):
            batch_eop_pair_cos_sim[i] = torch.cat((cos_sim, (torch.ones(max_eop_sent_cnt - cos_sim.shape[0]) * -100).to(cos_sim.device))).unsqueeze(0)      # shape is 1 * max_eop_sent_cnt. -100 if for filtering return values of predict
            batch_eop_labels[i] = torch.cat((eop_labels, (torch.ones(max_eop_sent_cnt - eop_labels.shape[0], dtype=eop_labels.dtype) * -100).to(eop_labels.device))).unsqueeze(0)
        batch_eop_pair_cos_sim = torch.cat(batch_eop_pair_cos_sim)        # shape is (bs, max_eop_sent_cnt)
        batch_eop_labels = torch.cat(batch_eop_labels)
        
        return batch_eop_pair_cos_sim, batch_eop_labels


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''
    # https://amaarora.github.io/2020/06/29/FocalLoss.html
    # https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
    def __init__(self, gamma, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none', label_smoothing=label_smoothing)
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        loss = None
        cross_entropy = super().forward(input_, target)
        if torch.isnan(cross_entropy).any():
            print("cross_entropy: ", cross_entropy)
            loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(input_.device)
            return loss
        
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy

        # print("cross_entropy.mean(): ", cross_entropy.mean(), "focal_loss.mean(): ", loss.mean(), "cross_entropy.sum(): ", cross_entropy.sum())
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


def get_loss_fct(gamma, weight_label_zero, device):
    weight = None
    if weight_label_zero != 0.5:
        weight = torch.tensor([weight_label_zero, 1 - weight_label_zero], dtype=torch.float32).to(device)
    
    if gamma != 0:
        loss_fct = FocalLoss(gamma=gamma, weight=weight)
    else:
        loss_fct = CrossEntropyLoss(weight=weight)
    return loss_fct


def multiple2one_pooling(src, index, dim=1, pool_type="amax"):
    # shape of result is the same as src by padding empty tensor
    return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

