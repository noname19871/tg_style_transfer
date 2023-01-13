from typing import Optional

import torch.cuda
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image


class VGGStyleTransfer(nn.Module):
    def __init__(self, device: Optional[str] = None):
        super().__init__()

        if device is None:
            self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device: str = device

        self.image_size: int = 512 if self.device == 'cuda' else 128
        self.image_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()

    def load_image(self, image_path):
        image = Image.open(image_path)
        # fake batch dimension required to fit network's input dimensions
        image = self.image_transforms(image).unsqueeze(0)
        return image.to(self.device, torch.float)