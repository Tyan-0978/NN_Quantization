# ------------------------------------------------------------------------------
# experiment of partial sum bitwidth
# ------------------------------------------------------------------------------

import os
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from tqdm.auto import tqdm

from utils import helper
from utils import alexnet_utils
from utils import psum


# set partial sum parameters
psum_bits = 32
psum_num_ops = 2

# create custom AlexNet model
model = alexnet_utils.create_alexnet_model()
custom_model = psum.CustomAlexnet(
    ref_model=model,
    psum_bits=psum_bits,
    psum_num_ops=psum_num_ops,
    )

# load quantization parameters and setup custom model
params_dir = 'params/'
params_name = 'alexnet_imagenet_qtz_histogram_acc53_params.pkl'
params_path = os.path.join(params_dir, params_name)

with open(params_path, 'rb') as file:
    params = pickle.load(file)

custom_model.customize(params)

#print(custom_model)

# dataset
num_data = 10

transform = alexnet_utils.use_alexnet_transform()
test_set = datasets.ImageNet(root="./data", split='val', transform=transform)
test_subset = Subset(test_set, list(range(num_data)))
test_loader = DataLoader(test_subset)

'''
for inputs, labels in tqdm(test_loader):
    model_out = model(inputs)
    custom_out = custom_model(inputs)
'''

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

topk = (1, 5)
with torch.no_grad():
    _, fp32_top1, fp32_top5 = helper.evaluate_model_topk(
        model, test_loader,
        device=gpu_device,
        topk=topk,
        )
    _, custom_top1, custom_top5 = helper.evaluate_model_topk(
        custom_model, test_loader,
        device=cpu_device,
        topk=topk,
        )

print(f'Original model accuracy: {fp32_top1:.5f} (top 1) / {fp32_top5:.5f} (top 5)')
print(f'Custom model accuracy:   {fp32_top1:.5f} (top 1) / {fp32_top5:.5f} (top 5)')
