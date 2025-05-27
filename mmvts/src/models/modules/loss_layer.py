
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .contrastive_learning_layer import *


class LossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.modality_cl_module = ModalityContrastiveLearning(config)
        self.topic_mm_cl_module = TopicContrastiveLearning(config)

    def compute_ts_loss(self, labels, logits):
        selected_labels = labels[labels != -100]
        loss_fct = CrossEntropyLoss()
        weight_label_zero = self.config.weight_label_zero
        if weight_label_zero != 0.5:
            weight = torch.tensor([weight_label_zero, 1 - weight_label_zero], dtype=torch.float32).to(labels.device)
            loss_fct = CrossEntropyLoss(weight=weight)

        ts_loss = self.config.ts_lw * loss_fct(logits.reshape(-1, self.config.num_labels), selected_labels.reshape(-1))
        return ts_loss
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def get_modality_cl_loss(self, text_feature=None, visual_feature=None, audio_feature=None):
        av_cl_loss, at_cl_loss, tv_cl_loss = None, None, None
        if self.config.fuse_type == "cat_a_t":
            modality_cl_loss = self.config.modality_cl_lw * self.modality_cl_module(audio_feature, text_feature)
        elif self.config.fuse_type == "cat_a_v":
            modality_cl_loss = self.config.modality_cl_lw * self.modality_cl_module(audio_feature, visual_feature)
        elif self.config.fuse_type == "cat_t_v":
            modality_cl_loss = self.config.modality_cl_lw * self.modality_cl_module(text_feature, visual_feature)
        elif self.config.fuse_type in ["cat", "mean", "max"]:
            modality_cl_loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(text_feature.device)
            if self.config.do_align_av:
                av_cl_loss = self.config.align_av_weight * self.modality_cl_module(audio_feature, visual_feature)
                modality_cl_loss += av_cl_loss
            if self.config.do_align_at:
                at_cl_loss = self.config.align_at_weight * self.modality_cl_module(audio_feature, text_feature)
                modality_cl_loss += at_cl_loss
            if self.config.do_align_tv:
                tv_cl_loss = self.config.align_tv_weight * self.modality_cl_module(text_feature, visual_feature)
                modality_cl_loss += tv_cl_loss
            modality_cl_loss = self.config.modality_cl_lw * modality_cl_loss
        else:
            raise ValueError("not supported modality cl loss for fuse_type {}".format(self.config.fuse_type))
        
        return modality_cl_loss, av_cl_loss, at_cl_loss, tv_cl_loss

    def forward(self, 
                labels, 
                logits, 
                text_features=None,
                visual_features=None,
                audio_features=None,
                fused_features=None, 
                moe_loss=None, 
                projected_text_features=None, 
                projected_visual_features=None, 
                projected_audio_features=None
            ):
        # labels 有batch维度（对比学习需要构造正负样本，防止sample的最后一个topic和下一个sample的第一个topic混为同一个topic），其余没有batch维度
        loss_dict = {}
        total_loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(labels.device)

        # topic segmentation loss
        ts_loss = self.compute_ts_loss(labels, logits)
        loss_dict["ts_loss"] = ts_loss
        total_loss = ts_loss
        
        # contrastive learning loss
        if self.config.do_modality_cl:
            if self.config.align_before_fuse:
                modality_cl_loss, av_cl_loss, at_cl_loss, tv_cl_loss = self.get_modality_cl_loss(projected_text_features, projected_visual_features, projected_audio_features)
            else:
                modality_cl_loss, av_cl_loss, at_cl_loss, tv_cl_loss = self.get_modality_cl_loss(text_features, visual_features, audio_features)
            print("modality_cl_loss: ", modality_cl_loss)
            loss_dict["modality_cl_loss"] = modality_cl_loss

            if av_cl_loss is not None:
                loss_dict["av_cl_loss"] = av_cl_loss
            if at_cl_loss is not None:
                loss_dict["at_cl_loss"] = at_cl_loss
            if tv_cl_loss is not None:
                loss_dict["tv_cl_loss"] = tv_cl_loss

            total_loss = total_loss + modality_cl_loss

        if self.config.do_topic_mm_cl:
            topic_mm_cl_loss = self.config.topic_mm_cl_lw * self.topic_mm_cl_module(fused_features, labels, self.config.topic_mm_cl_pos_k, self.config.topic_mm_cl_neg_k)
            print("topic_mm_cl_loss: ", topic_mm_cl_loss)
            loss_dict["topic_mm_cl_loss"] = topic_mm_cl_loss
            total_loss = total_loss + topic_mm_cl_loss

        if moe_loss is not None:
            print("moe_loss: ", moe_loss)
            loss_dict["moe_loss"] = moe_loss
            total_loss = total_loss + moe_loss
        
        loss_dict["total_loss"] = total_loss
        return {"loss": loss_dict}
        