
import os
import glob
import json

import torch
import torch.nn as nn

from .vis2d_encoder import Vis2dEncoder
from .vis3d_encoder import Vis3dEncoder
from .vis_ocr_encoder import VisOcrEncoder


class VisEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 提取各类的 vis feature，cat之后进行线性变换得到和与 text feature 同样维度的张量
        self.config = config
        
        self.vis2d_encoder = Vis2dEncoder(config)
        self.vis3d_encoder = Vis3dEncoder(config)
        self.visocr_encoder = VisOcrEncoder(config)

        # self.load_data()

        if self.config.freeze_vis2d_encoder:
            self.freeze_vis2d_encoder()
    
    def freeze_vis2d_encoder(self):
        print("freeze_vis2d_encoder is True.")
        for param in self.vis2d_encoder.parameters():
            param.requires_grad = False
    
    def extract_vis2d_embeds(self, current_mode, example_ids, vis_ids):
        batch_vis2d_features = []
        for example_index, example_vis_ids in zip(example_ids, vis_ids):
            example_vis2d_features = []
            example_info = self.data[current_mode][example_index.item()]
            for vis_id in example_vis_ids:
                if vis_id != -1:
                    vis_stet = example_info["example_stet"][vis_id.item()]
                    vis_start, vis_end = int(vis_stet[0]), int(vis_stet[1])
                    vis_paths = [self.tmpl.format(example_info["example_frame_folder"], second, second) for second in range(vis_start, vis_end)]
                    vis_paths = [v for v in vis_paths if v in example_info["example_frame_paths"]]
                    if vis_paths:
                        vis2d_features = self.vis2d_encoder(vis_paths, vis_ids.device)        # shape is (vis_end - vis_start, hidden_size)
                        vis2d_features = torch.max(vis2d_features, 0)[0]
                    else:
                        vis2d_features = torch.zeros((self.config.hidden_size_vis2d)).to(vis_ids.device)
                else:
                    vis2d_features = torch.zeros((self.config.hidden_size_vis2d)).to(vis_ids.device)
                # print("vis2d_features.shape: ", vis2d_features.shape)
                example_vis2d_features.append(vis2d_features.unsqueeze(0))
            example_vis2d_features = torch.cat(example_vis2d_features, dim=0)
            # print("example_vis2d_features.shape: ", example_vis2d_features.shape)
            batch_vis2d_features.append(example_vis2d_features.unsqueeze(0))

        batch_vis2d_features = torch.cat(batch_vis2d_features, dim=0)
        # print("batch_vis2d_features.shape: ", batch_vis2d_features.shape)
        return batch_vis2d_features

    def extract_vis3d_embeds(self, current_mode, example_ids, vis_ids):
        pass

    def extract_vis_ocr_embeds(self, current_mode, example_ids, vis_ids):
        pass
        
    def forward(
        self, 
        example_ids, 
        vis_ids, 
        vis2d_embeds=None, 
        vis3d_embeds=None, 
        vis_ocr_embeds=None,
    ):
        if not self.config.use_vis2d and not self.config.use_vis3d and not self.config.use_vis_ocr:
            return None

        if not self.config.freeze_vis2d_encoder or vis2d_embeds is None:
            vis2d_embeds = self.extract_vis2d_embeds(current_mode, example_ids, vis_ids)
        if vis3d_embeds is None and self.config.use_vis3d:
            vis3d_embeds = self.extract_vis3d_embeds(current_mode, example_ids, vis_ids)
        if vis_ocr_embeds is None and self.config.use_vis_ocr:
            vis_ocr_embeds = self.extract_vis_ocr_embeds(current_mode, example_ids, vis_ids)

        features_to_cat = []

        if self.config.use_vis2d:
            features_to_cat.append(vis2d_embeds)
        if self.config.use_vis3d:
            features_to_cat.append(vis3d_embeds)
        if self.config.use_vis_ocr:
            features_to_cat.append(vis_ocr_embeds)
        
        vis_feature = torch.cat(features_to_cat, dim=-1)
        return vis_feature
