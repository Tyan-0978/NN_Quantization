# ------------------------------------------------------------------------------
# utilities for AlexNet
# ------------------------------------------------------------------------------

import torch
import torchvision
from torchvision import transforms
from torchvision.models.alexnet import alexnet, AlexNet_Weights

def create_alexnet_model():
  alexnet_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
  return alexnet_model

def use_alexnet_transform():
  mean = [0.485, 0.456, 0.406]
  std_dev = [0.229, 0.224, 0.225]

  alexnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std_dev)
  ])

  return alexnet_transform
