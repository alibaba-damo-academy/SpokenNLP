
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from .utils import Similarity


class CSSL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cos_sim_fct = Similarity(temp=self.config.cl_temp)
    
    def multiple2one_pooling(self, src, index, dim=1, pool_type="amax"):
        # shape of result is the same as src by padding empty tensor
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)

    def eop_level_matrix_cl_loss(
        self,
        anchor_eop_features,
        cl_segment_ids,
        ):
        '''
        eg. anchor_eop_features is [e1t1p1, e1t1p2, e1t1p3, e1t2p1, e1t3p1, e1t3p2, ...] where e means example, t means topic, p means paragraph
        we can construct similarity matrix like this:
            a1,     a2, a3, b1, c1, c2
        a1  ignore,  +,  +,  -,  -,  -
        a2  +     ignore +,  -,  -,  -
        a3  +,     +, ignore,-,  -,  -
        b1  -,      -,  -,  ignore, -,
        c1
        c2
        '''
        loss = None
        # get each eop sentence feature, and compute n * n cosine similarity matrix, then numerator is all positive eop pairs and denominator is all positive and negative eop pairs
        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        d_cnt = {i: cl_segment_ids.count(i) for i in range(total_topic_cnt)}    # {0:3, 1:1, 2:2, 3:1, 4:2}
        d_left = {i: accumulate_eop_cnts[i] for i in range(total_topic_cnt)}
        d_right = {i: total_eop_cnt - d_cnt[i] - d_left[i] for i in range(total_topic_cnt)}

        sim_mask_for_numerator, sim_mask_for_denominator = [], []
        for i, id_ in enumerate(cl_segment_ids):
            sim_mask = [False] * d_left[id_] + [True] * d_cnt[id_] + [False] * d_right[id_]
            sim_mask[i] = False
            sim_mask_for_numerator.append(sim_mask)
            sim_mask = [True] * d_left[id_] + [False] * d_cnt[id_] + [True] * d_right[id_]
            sim_mask_for_denominator.append(sim_mask)
        
        sim_mask_for_numerator = torch.tensor(sim_mask_for_numerator).to(anchor_eop_features.device)
        sim_mask_for_denominator = torch.tensor(sim_mask_for_denominator).to(anchor_eop_features.device)

        # get st features from anchor_eop_features self
        st_cos_sim = self.cos_sim_fct(anchor_eop_features.unsqueeze(1), anchor_eop_features.unsqueeze(0))
        exp_st_cos_sim = torch.exp(st_cos_sim)
        
        numerator = torch.sum(sim_mask_for_numerator * exp_st_cos_sim, 0)
        denominator = numerator + torch.sum(sim_mask_for_denominator * exp_st_cos_sim, 0)

        cl_prob = numerator / denominator
        if torch.isnan(cl_prob).any():
            print("denominator is zero!")
        elif min(cl_prob[cl_prob != 0].shape) == 0:
            print("cl_prob[cl_prob!=0] is empty")
        else:
            cl_loss = -1 * torch.log(cl_prob[cl_prob != 0])
            loss = cl_loss.mean()
        
        return loss

    def eot_level_matrix_cl_loss(
        self,
        anchor_eop_features,
        cl_segment_ids,
    ):
        pass
    
    def cl_loss_for_list(
        self,
        anchor_eop_features,    # all eop features
        anchor_features,        # anchor features for compute cl loss
        positive_eop_indices,   # (cl_positive_k, anchor_features.shape[0])
        negative_eop_indices,
    ):
        loss = torch.tensor(0.0).to(anchor_eop_features.device)
        # compute cl loss by similarity
        m_positive_similarity = []
        for i in range(self.config.cl_positive_k):
            positive_eop_features = anchor_eop_features[positive_eop_indices[i]]
            positive_similarity = self.cos_sim_fct(anchor_features, positive_eop_features)
            m_positive_similarity.append(positive_similarity.unsqueeze(0))
        m_positive_similarity = torch.cat(m_positive_similarity).to(anchor_eop_features.device)        # shape is (self.config.cl_positive_k, total_topic_cnt)

        n_negative_similarity = []
        for i in range(self.config.cl_negative_k):
            negative_eop_features = anchor_eop_features[negative_eop_indices[i]]
            negative_similarity = self.cos_sim_fct(anchor_features, negative_eop_features)
            n_negative_similarity.append(negative_similarity.unsqueeze(0))
        n_negative_similarity = torch.cat(n_negative_similarity).to(anchor_eop_features.device)        # shape is (self.config.cl_negative_k, total_topic_cnt)

        similarity_matrix = torch.cat((m_positive_similarity, n_negative_similarity))   # shape is (self.config.cl_positive_k + self.config.cl_negative_k, total_topic_cnt)
        exp_similarity_matrix = torch.exp(similarity_matrix)

        sim_mask_for_numerator = [[1] * anchor_features.shape[0] for _ in range(self.config.cl_positive_k)] + [[0] * anchor_features.shape[0] for _ in range(self.config.cl_negative_k)]
        sim_mask_for_numerator = torch.tensor(sim_mask_for_numerator).to(anchor_eop_features.device)

        numerator = torch.sum(exp_similarity_matrix * sim_mask_for_numerator, 0)
        denominator = torch.sum(exp_similarity_matrix, 0)
        cl_loss = -1 * torch.log(numerator / denominator)
        loss = cl_loss.mean()
        
        return loss

    def eop_level_list_cl_loss(
        self,
        anchor_eop_features,
        cl_segment_ids,
    ):
        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]

        bot_indices = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        eot_indices = [v - 1 for v in accumulate_eop_cnts[1:]] + [total_eop_cnt - 1]
        
        topic_id2bot_eot_indices = {}
        for id_, (start_id, end_id) in enumerate(zip(bot_indices, eot_indices)):
            topic_id2bot_eot_indices[id_] = (start_id, end_id)

        positive_eop_indices = [[] for _ in range(self.config.cl_positive_k)]
        negative_eop_indices = [[] for _ in range(self.config.cl_negative_k)]
        for eop_index, eop_topic_id in enumerate(cl_segment_ids):
            start_id, end_id = topic_id2bot_eot_indices[eop_topic_id]

            choice_ids = list(range(start_id, end_id))
            if len(choice_ids) == 0:
                choice_ids = [end_id]
            pos_id = eop_index
            for i in range(self.config.cl_positive_k):
                pos_id -= 1
                if pos_id < start_id:
                    # print("choice_ids: ", choice_ids)
                    pos_id = random.choice(choice_ids)
                positive_eop_indices[i].append(pos_id)
            
            choice_ids = list(range(end_id + 1, eot_indices[-1] + 1))
            if len(choice_ids) == 0:
                choice_ids = list(range(bot_indices[0], bot_indices[1]))
            pos_id = end_id
            for i in range(self.config.cl_negative_k):
                pos_id += 1
                if pos_id >= total_eop_cnt:
                    # print("choice_ids: ", choice_ids)
                    pos_id = random.choice(choice_ids)
                negative_eop_indices[i].append(pos_id)

        loss = self.cl_loss_for_list(
            anchor_eop_features,
            anchor_eop_features,
            positive_eop_indices,
            negative_eop_indices,
            )
        return loss

    def eot_level_list_cl_loss(
        self, 
        anchor_eop_features,
        cl_segment_ids,
        ):
        '''
        get cl_positive_k and cl_negative_k eop features for anchor eot features,
        then compute similarity of each pair consists of eot and one eop.
        eg. a1 a2 a3 b1 c1 c2
        first anchor eot features are [a3, b1, c2]
        if cl_positive_k == 1, then positive eop features are [a2, b1, c1]
        if cl_negative_k == 1, then negative eop features are [b1, c1, a1].
        we can implement the idea by cl_segment_ids [0 0 0 1 2 2]
        '''       
        loss = None

        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]

        # anchor eot indices and features
        bot_indices = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        eot_indices = [v - 1 for v in accumulate_eop_cnts[1:]] + [total_eop_cnt - 1]
        eot_features = anchor_eop_features[eot_indices]

        # positive eop indices and features
        positive_eop_indices = [[] for _ in range(self.config.cl_positive_k)]       # each value is a list whose length is total_topic_cnt
        for start_id, end_id in zip(bot_indices, eot_indices):
            choice_ids = list(range(start_id, end_id))
            if len(choice_ids) == 0:
                choice_ids = [end_id]
            pos_id = end_id
            for i in range(self.config.cl_positive_k):
                pos_id -= 1
                if pos_id < start_id:
                    # print("choice_ids: ", choice_ids)
                    pos_id = random.choice(choice_ids)
                positive_eop_indices[i].append(pos_id)

        # negative eop indices and features
        negative_eop_indices = [[] for _ in range(self.config.cl_negative_k)]
        for end_id in eot_indices:
            choice_ids = list(range(end_id + 1, eot_indices[-1] + 1))
            if len(choice_ids) == 0:
                choice_ids = list(range(bot_indices[0], bot_indices[1]))
            pos_id = end_id
            for i in range(self.config.cl_negative_k):
                pos_id += 1
                if pos_id >= total_eop_cnt:
                    # print("choice_ids: ", choice_ids)
                    pos_id = random.choice(choice_ids)
                negative_eop_indices[i].append(pos_id)

        loss = self.cl_loss_for_list(
            anchor_eop_features,
            eot_features,
            positive_eop_indices,
            negative_eop_indices,
            )
        return loss

    def forward(
        self,
        sequence_output,
        labels,
        extract_eop_segment_ids,
        eop_index_for_aggregate_batch_eop_features,
        ):

        loss = torch.tensor(0.0).to(sequence_output.device)
        bs, seq_length, hidden_size = sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]

        # compute contrastive learning loss
        eop_level_output = self.multiple2one_pooling(sequence_output, extract_eop_segment_ids, pool_type="amax")    # contains cls, shape is (bs, seq_len, hidden_size)
        tmp_eop_index = eop_index_for_aggregate_batch_eop_features + torch.arange(bs).to(sequence_output.device).unsqueeze(1).expand_as(eop_index_for_aggregate_batch_eop_features) * seq_length

        tmp_eop_index = tmp_eop_index.reshape(-1)
        eop_index_for_aggregate_batch_eop_features = eop_index_for_aggregate_batch_eop_features.reshape(-1)
        eop_index = tmp_eop_index[eop_index_for_aggregate_batch_eop_features != 0]
        anchor_eop_features = eop_level_output.reshape(bs * seq_length, -1)[eop_index]

        eop_labels = [l[l != -100] for l in labels]
        cl_segment_ids = []     # for recording topic level segment id. [a1, a2, a3, b1, c1, c2] will have cl_segment_ids [0, 0, 0, 1, 2, 2]
        seg_id = 0
        for example_eop_labels in eop_labels:
            if len(example_eop_labels) == 0:
                continue
            for l in example_eop_labels:
                cl_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_eop_labels[-1] == 1:
                seg_id += 1

        if len(cl_segment_ids) > 2 and cl_segment_ids[-1] > 0:
            # must have at least 2 topic
            if self.config.cl_anchor_level == "eop_matrix":
                loss = self.eop_level_matrix_cl_loss(anchor_eop_features, cl_segment_ids)
            elif self.config.cl_anchor_level == "eot_list":
                loss = self.eot_level_list_cl_loss(anchor_eop_features, cl_segment_ids)
            elif self.config.cl_anchor_level == "eop_list":
                loss = self.eop_level_list_cl_loss(anchor_eop_features, cl_segment_ids)
            else:
                raise ValueError("not supported cl_anchor_level %s " % self.config.cl_anchor_level)
        
        return loss
