
import torch
import torch.nn as nn
import torch.nn.functional as F

import random


eps = 1e-8
class Similarity(nn.Module):
    """
    cosine similarity
    """
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, x, y):
        x = x / (x.norm(dim=1, keepdim=True) + eps)
        y = y / (y.norm(dim=1, keepdim=True) + eps)
        if self.temp == 0:
            return torch.matmul(x, y.t())
        else:
            return torch.matmul(x, y.t()) / self.temp


class ModalityContrastiveLearning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cos_sim_fct = Similarity(temp=self.config.cl_temp)
    
    def forward(self, modality1, modality2):
        similarity_matrix = self.cos_sim_fct(modality1, modality2)
        numerator = torch.exp(torch.diag(similarity_matrix)) + eps
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1) + eps
        contrastive_loss = -torch.log(numerator / denominator).mean()
        return contrastive_loss


class TopicContrastiveLearning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cos_sim_fct = Similarity(temp=self.config.cl_temp)
    
    def get_valid_labels(self, labels):
        # labels = [[-100, 0, 1, -100, 0], [-100, 0, 0, -100, 1]]
        # valid_labels is [0, 1, 1, 0, 0, 1]
        mask = labels != -100
        valid_features_count = mask.sum(1)
        # seq_length = valid_features_count.sum()
        # 需要将每个sample的最后一个label置为1，防止该sample的最后一个topic和下一个sample的第一个topic混为1个topic
        sample_last_label_indices = torch.cumsum(valid_features_count, dim=0) - 1

        valid_labels = labels[labels != -100]
        valid_labels[sample_last_label_indices] = 1
        return valid_labels
        
    def get_chunk_mask_for_matrix_loss(self, valid_labels):
        '''
        valid_labels is [0, 1, 1, 0, 0, 1]
        valid_mask is 
        [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
        ]
        '''
        seq_length = valid_labels.shape[0]
        # 初始化 mask 矩阵为 0
        valid_mask = torch.zeros((seq_length, seq_length), dtype=torch.int)
        # 找到所有值为 1 的下标
        ones_indices = (valid_labels == 1).nonzero(as_tuple=False).squeeze(0)
        # 根据值为 1 的下标创建 chunks
        start_idx = 0
        for idx in ones_indices:
            end_idx = idx
            valid_mask[start_idx:end_idx+1, start_idx:end_idx+1] = 1
            start_idx = end_idx + 1

        diagonal_mask = ~torch.eye(seq_length, dtype=torch.bool)
        valid_mask = valid_mask & diagonal_mask
        return valid_mask.to(valid_labels.device)

    def matrix_type_loss(self, seq_features, valid_labels):
        valid_mask = self.get_chunk_mask_for_matrix_loss(valid_labels)
        similarity_matrix = self.cos_sim_fct(seq_features, seq_features)
        exp_sim_matrix = torch.exp(similarity_matrix)
        # 计算分子：每一行 chunk 内的 feature 对的相似度
        numerator = torch.sum(exp_sim_matrix * valid_mask, dim=1) + eps
        # 创建一个掩码来计算分母，排除对角线元素
        valid_denominator_mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.shape[0]).to(valid_labels.device)
        # 计算分母：所有 feature 对的相似度，排除对角线
        denominator = torch.sum(exp_sim_matrix * valid_denominator_mask, dim=1) + eps
        # 计算对比损失
        contrastive_loss = -torch.log(numerator / denominator).mean()
        return contrastive_loss

    def locate_topics(self, valid_labels):
        # 定位每个topic的起止下标: [0, 1, 1, 0, 0, 1] -> {0: (0, 1), 1: (2, 2), 2: (3, 5)}
        cl_segment_ids = []
        seg_id = 0
        for l in valid_labels:
            cl_segment_ids.append(seg_id)
            if l == self.config.label_eot:
                seg_id += 1
        # print("valid_labels: {}, cl_segment_ids: {}".format(valid_labels, cl_segment_ids))

        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]

        bot_indices = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        eot_indices = [v for v in accumulate_eop_cnts[1:]] + [total_eop_cnt]

        topic_id2bot_eot_indices = {}
        for id_, (start_id, end_id) in enumerate(zip(bot_indices, eot_indices)):
            topic_id2bot_eot_indices[id_] = (start_id, end_id)  # 左闭右开 [)
        
        # print("bot_indices: {}, eot_indices: {}, topic_id2bot_eot_indices: {}".format(bot_indices, eot_indices, topic_id2bot_eot_indices))
        return topic_id2bot_eot_indices

    def anchor_cl_loss(self, anchor_feature, pos_features, neg_features):
        # print("anchor_feature.shape: ", anchor_feature.shape)
        # print("pos_features.shape: ", pos_features.shape)
        # print("neg_features.shape: ", neg_features.shape)
        if self.config.topic_cl_fct == "ce":
            # 计算正样本和负样本与锚点之间的相似度
            pos_sim = F.cosine_similarity(anchor_feature, pos_features)
            neg_sim = F.cosine_similarity(anchor_feature, neg_features)
            # print("pos_sim: {}, neg_sim: {}".format(pos_sim, neg_sim))

            # 合并正负样本相似度
            all_sim = torch.cat([pos_sim, neg_sim], dim=0)
            
            # 创建对应的标签，正样本为1，负样本为0
            labels = torch.cat([torch.ones(pos_sim.size(0)), torch.zeros(neg_sim.size(0))], dim=0).to(anchor_feature.device)

            # 使用二元交叉熵损失计算对比损失
            loss = F.binary_cross_entropy_with_logits(all_sim, labels)
        elif self.config.topic_cl_fct == "simcse":
            # 计算anchor与所有样本的相似度
            pos_sim = self.cos_sim_fct(anchor_feature, pos_features)
            neg_sim = self.cos_sim_fct(anchor_feature, neg_features)

            # 计算相似度的指数
            pos_sim_exp = torch.exp(pos_sim)
            neg_sim_exp = torch.exp(neg_sim)

            # 计算分子（所有正样本对相似度的和）
            numerator = pos_sim_exp.sum()
            # 计算分母（所有样本对相似度的和）
            denominator = pos_sim_exp.sum() + neg_sim_exp.sum()

            # 计算最终的比值
            loss = -1.0 * torch.log(numerator / denominator)
        else:
            raise ValueError("not support topic_cl_fct {}".format(self.config.topic_cl_fct))
        return loss

    def select_pos_features(self, seq_features, topics, anchor_topic_idx, anchor_clip_idx, pos_k):
        topic_start = topics[anchor_topic_idx][0]
        topic_end = topics[anchor_topic_idx][1]

        # print("anchor_topic_idx: {}, topic_start: {}, topic_end: {}, anchor_clip_idx: {}, topics: {}".format(anchor_topic_idx, topic_start, topic_end, anchor_clip_idx, topics))
        
        if self.config.topic_cl_choice == "random":
            # 正样本将是同一主题内的其他特征
            choice_indices = list(range(topic_start, anchor_clip_idx)) + list(range(anchor_clip_idx + 1, topic_end))
            # print("choice_indices: ", choice_indices)

            # 确保有足够的正样本
            if len(choice_indices) < pos_k:
                add_cnt = pos_k - len(choice_indices)
                for _ in range(add_cnt):
                    choice_indices.append(random.choice(choice_indices))

            # 随机选择pos_k个正样本
            selected_pos_indices = torch.randperm(len(choice_indices))[:pos_k]
            pos_features = seq_features[[choice_indices[i] for i in selected_pos_indices]]
        elif self.config.topic_cl_choice == "near":
            # 正样本将是同一主题内的其他特征，按照距离选择
            choice_indices_left = list(range(anchor_clip_idx - 1, topic_start - 1, -1))
            choice_indices_right = list(range(anchor_clip_idx + 1, topic_end))

            def merge_arrays(nums1, nums2):
                res = []
                for a, b in zip(nums1, nums2):
                    res += [a, b]
                if len(nums1) < len(nums2):
                    res += nums2[len(nums1):]
                else:
                    res += nums1[len(nums2):]
                return res
            
            choice_indices = merge_arrays(choice_indices_left, choice_indices_right)
            # print("choice_indices_left: {}, choice_indices_right: {}, choice_indices: {}".format(choice_indices_left, choice_indices_right, choice_indices))

            selected_pos_indices = []
            choice_index = 0
            for i in range(pos_k):
                if choice_index >= len(choice_indices):
                    selected_pos_indices.append(random.choice(choice_indices))
                else:
                    selected_pos_indices.append(choice_indices[choice_index])
            pos_features = seq_features[[pos_index for pos_index in selected_pos_indices]]
        else:
            raise ValueError("not support now")
            
        return pos_features

    def select_neg_features(self, seq_features, topics, anchor_topic_idx, neg_k):
        total_topic_cnt = len(topics)

        # print("anchor_topic_idx: {}, total_topic_cnt: {}, topics: {}".format(anchor_topic_idx, total_topic_cnt, topics))
        if self.config.topic_cl_choice == "random":
            # 随机选择neg个来自不同主题的特征作为负样本
            choice_indices = []
            for idx, (neg_start, neg_end) in topics.items():
                # 跳过当前主题
                if idx == anchor_topic_idx: 
                    continue
                choice_indices.extend(list(range(neg_start, neg_end)))
                
            # 确保有足够的负样本
            if len(choice_indices) < neg_k:
                add_cnt = neg_k - len(choice_indices)
                for _ in range(add_cnt):
                    choice_indices.append(random.choice(choice_indices))

            # 随机选择neg_k个负样本
            selected_neg_indices = torch.randperm(len(choice_indices))[:neg_k]
            neg_features = seq_features[[choice_indices[i] for i in selected_neg_indices]]
        elif self.config.topic_cl_choice == "near":
            # 按照距离选择来自不同主题的特征作为负样本

            if anchor_topic_idx < total_topic_cnt - 1:
                # 不是最后一个topic,候选特征的位置是下一个topic的start到最后一个topic的end
                choice_indices = list(range(topics[anchor_topic_idx + 1][0], topics[total_topic_cnt - 1][1]))
            else:
                # 是最后一个topic,候选特征的位置是倒数第二个topic的end到第一个topic的start
                choice_indices = list(range(topics[anchor_topic_idx - 1][1], topics[0][0], -1))

            selected_neg_indices = []
            choice_index = 0
            for i in range(neg_k):
                if choice_index >= len(choice_indices): 
                    selected_neg_indices.append(random.choice(choice_indices))
                else:
                    selected_neg_indices.append(choice_indices[choice_index])
                choice_index += 1
            neg_features = seq_features[[neg_index for neg_index in selected_neg_indices]]
        else:
            raise ValueError("not support now")

        return neg_features
        
    def list_type_loss(self, seq_features, valid_labels, pos_k, neg_k):
        topics = self.locate_topics(valid_labels)
        total_loss = torch.tensor(0.0).to(seq_features.device)
        if len(topics) == 1:
            return total_loss

        loss_num = 0
        for anchor_topic_idx, (start, end) in topics.items():
            if start + 1 == end:
                # topic仅包含1个clip
                continue
            # 锚点特征在当前主题范围内进行迭代
            for anchor_clip_idx in range(start, end):
                anchor_feature = seq_features[anchor_clip_idx].unsqueeze(0) # 增加维度以便广播
                pos_features = self.select_pos_features(seq_features, topics, anchor_topic_idx, anchor_clip_idx, pos_k)
                neg_features = self.select_neg_features(seq_features, topics, anchor_topic_idx, neg_k)   
                # 计算锚点特征对应的对比损失
                total_loss = total_loss + self.anchor_cl_loss(anchor_feature, pos_features, neg_features)
                loss_num += 1
        
        if loss_num != 0:
            total_loss = total_loss / loss_num

        return total_loss 

    def forward(self, seq_features, labels, pos_k, neg_k):
        valid_labels = self.get_valid_labels(labels)
        if self.config.topic_cl_type == "matrix":
            contrastive_loss = self.matrix_type_loss(seq_features, valid_labels)
        elif self.config.topic_cl_type == "list":
            contrastive_loss = self.list_type_loss(seq_features, valid_labels, pos_k, neg_k)
        else:
            raise ValueError("not supported topic_cl_type: {}".format( self.config.topic_cl_type))
        return contrastive_loss
