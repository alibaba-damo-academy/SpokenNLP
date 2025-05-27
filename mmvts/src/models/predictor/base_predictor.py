
import torch
import torch.nn as nn

from ..modules.loss_layer import LossLayer


class BasePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_layer = LossLayer(config)

    def get_logits(self, fused_features):
        raise NotImplementedError("get_logits function is not implemented")
    
    
    def fuse_features(self, text_features=None, visual_features=None, audio_features=None):
        # labels 有batch维度，其余没有batch维度
        if self.config.fuse_type == "text_only":
            assert text_features is not None
            fused_features = text_features
        elif self.config.fuse_type == "vis_only":
            assert visual_features is not None
            fused_features = visual_features
        elif self.config.fuse_type == "audio_only":
            assert audio_features is not None
            fused_features = audio_features
        elif self.config.fuse_type == "cat_a_t":
            fused_features = torch.cat((audio_features, text_features), dim=1) 
        elif self.config.fuse_type == "cat_a_v":
            fused_features = torch.cat((audio_features, visual_features), dim=1) 
        elif self.config.fuse_type == "cat_t_v":
            fused_features = torch.cat((text_features, visual_features), dim=1) 
        elif self.config.fuse_type == "cat":
            fused_features = torch.cat((text_features, visual_features, audio_features), dim=1)
        elif self.config.fuse_type == "mean":
            fused_features = torch.mean(torch.stack((text_features, visual_features, audio_features), dim=0), dim=0)
        elif self.config.fuse_type == "max":
            fused_features = torch.max(torch.stack((text_features, visual_features, audio_features), dim=0), dim=0)[0]
        else:
            raise ValueError("not supported fuse_type: {}".format(self.config.fuse_type))
        
        return fused_features

    def forward(self, labels, text_features=None, visual_features=None, audio_features=None, moe_loss=None, projected_text_features=None, projected_visual_features=None, projected_audio_features=None):
        # # labels 有batch维度，其余没有batch维度
        # fused_features = self.fuse_features(text_features, visual_features, audio_features)
        # logits = self.get_logits(fused_features)    
        # loss_dict = self.loss_layer(
        #     labels=labels, 
        #     logits=logits,
        #     text_features=text_features,
        #     visual_features=visual_features,
        #     audio_features=audio_features,
        #     fused_features=fused_features,
        #     moe_loss=moe_loss,
        # )
        # return logits, loss_dict["loss"]["total_loss"]
        raise NotImplementedError("get_logits function is not implemented")