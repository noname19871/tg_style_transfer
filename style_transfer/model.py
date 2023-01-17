from typing import Optional

import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm

from PIL import Image
from torchvision.utils import save_image

from style_transfer.loss import ContentLoss, StyleLoss
from style_transfer.normalizer import Normalization


class VGGStyleTransfer(nn.Module):
    def __init__(self, device: Optional[str] = None):
        super().__init__()

        if device is None:
            self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device: str = device

        self.image_size: int = 512 if self.device == 'cuda' else 128
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        self.vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(self.device).eval()

    def build_model(self, content_image, style_image):
        # desired depth layers to compute style/content losses :
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        content_losses = []
        style_losses = []

        # use recommended mean and std
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        # normalization module
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(self.device)

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def run_style_transfer(self, content_image,
                           style_image,
                           image_path: str = None,
                           num_steps=100,
                           style_weight=1000000,
                           content_weight=1):
        model, style_losses, content_losses = self.build_model(content_image, style_image)

        # We want to optimize the input and not the model parameters, so we
        # update all the requires_grad fields accordingly
        content_image.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optimizer = optim.LBFGS([content_image])

        for _ in tqdm(range(num_steps)):

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    content_image.clamp_(0, 1)

                optimizer.zero_grad()
                model(content_image)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            content_image.clamp_(0, 1)

        if image_path:
            save_image(content_image[0], image_path)
        return content_image

    def load_image(self, image_path):
        image = Image.open(image_path)
        # fake batch dimension required to fit network's input dimensions
        image = self.image_transforms(image).unsqueeze(0)
        return image.to(self.device, torch.float)
