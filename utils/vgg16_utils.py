# ------------------------------------------------------------------------------
# Utility functions for VGG16
# ------------------------------------------------------------------------------

import torch
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

def create_vgg16_model():
  vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
  return vgg16_model

def use_vgg16_transform():
  vgg16_transform = VGG16_Weights.IMAGENET1K_V1.transforms()
  return vgg16_transform
