# ------------------------------------------------------------------------------
# experiment of partial sum bitwidth
# ------------------------------------------------------------------------------

import os
import sys
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from tqdm.auto import tqdm

from utils import helper
from utils import alexnet_utils
from utils import psum


# create models
# original AlexNet model
model = alexnet_utils.create_alexnet_model()

# quantized AlexNet model
model_dir = 'models/'
qtz_model_name = 'alexnet_imagenet_qtz_acc53_model.pt'
qtz_model_path = os.path.join(model_dir, qtz_model_name)
qtz_model = helper.load_torchscript_model(qtz_model_path)

# psum customized AlexNet model
psum_bits = 32
psum_num_ops = 36

custom_model = psum.CustomAlexnet(
    ref_model=model,
    psum_bits=psum_bits,
    psum_num_ops=psum_num_ops,
    )

# load quantization parameters and setup custom model
params_dir = 'params/'
params_name = 'alexnet_imagenet_qtz_acc53_params.pkl'
params_path = os.path.join(params_dir, params_name)

with open(params_path, 'rb') as file:
    params = pickle.load(file)

custom_model.customize(params)


# dataset
dataset_size = 50000
num_per_cat = 50
data_offset = 0
data_indices_1 = [i for i in range(data_offset, dataset_size, num_per_cat)]

num_data = 10000
data_offset = 40000
data_indices_2 = [i + data_offset for i in range(num_data)]

start = int(sys.argv[1]) - 1
end = int(sys.argv[2])
data_indices_3 = list(range(start, end))

print(f'Data range: {data_indices_3[0]} - {data_indices_3[-1]}')

transform = alexnet_utils.use_alexnet_transform()
test_set = datasets.ImageNet(root="./data", split='val', transform=transform)
test_subset = Subset(test_set, data_indices_3)
test_loader = DataLoader(test_subset)


# inference
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

topk = (1, 5)
with torch.no_grad():
    _, fp32_top1, fp32_top5 = helper.evaluate_model_topk(
        model, test_loader,
        device=gpu_device,
        topk=topk,
        )
    print(f'Original model accuracy: {fp32_top1:.5f} (top 1) / {fp32_top5:.5f} (top 5)')

    _, qtz_top1, qtz_top5 = helper.evaluate_model_topk(
        qtz_model, test_loader,
        device=cpu_device,
        topk=topk,
        )
    print(f'Quantized model accuracy: {qtz_top1:.5f} (top 1) / {qtz_top5:.5f} (top 5)')

    _, custom_top1, custom_top5 = helper.evaluate_model_topk(
        custom_model, test_loader,
        device=cpu_device,
        topk=topk,
        )
    print(f'Custom model accuracy: {custom_top1:.5f} (top 1) / {custom_top5:.5f} (top 5)')

