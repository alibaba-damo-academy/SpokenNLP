
import torch
import torch.nn as nn

from PIL import Image

from .clip import clip


vis2d_encoder_name_dict = {
    "pure_vit_b_16": "google/vit-base-patch16-224",
    "clip_vit_b_32": "ViT-B/32",
    "clip_vit_b_16": "ViT-B/16",
}
class Vis2dEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # if "clip" in self.config.vis2d_encoder_name:
        #     self.get_vis2d_encoder()
        
    def get_vis2d_encoder(self):
        # TODO: pure vit vs clip_vit
        # https://huggingface.co/google/vit-base-patch16-224
        model, preprocess = clip.load(vis2d_encoder_name_dict[self.config.vis2d_encoder_name])
        self.model = model
        self.preprocess = preprocess

    def forward(self, image_paths, device):
        images = [self.preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths if image_path]
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        images = torch.cat(images, dim=0).to(device)
        image_features = self.model.encode_image(images)
        return image_features
