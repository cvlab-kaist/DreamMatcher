import clip
import torch
from torchvision import transforms

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# class DINOdataset(torch.utils.data.Dataset):


class DINOEvaluator(object):
    def __init__(self, device) -> None:
        # self.device = device
        # self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
        # self.model = ViTModel.from_pretrained("facebook/dino-vits16")
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.model(img).detach()
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()
