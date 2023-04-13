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


if len(sys.argv) != 5:
    print(f'Missing arguments: Expecting 4 but get {len(sys.argv) - 1}')
    print(f'Arguments: {sys.argv}')
    sys.exit()

# create models
# original AlexNet model
model = alexnet_utils.create_alexnet_model()

# quantized AlexNet model
model_dir = 'models/'
qtz_model_name = 'alexnet_imagenet_qtz_acc53_model.pt'
qtz_model_path = os.path.join(model_dir, qtz_model_name)
qtz_model = helper.load_torchscript_model(qtz_model_path)

# psum customized AlexNet model
psum_bits = int(sys.argv[1])
psum_num_ops = int(sys.argv[2])
print(f'Psum bits: {psum_bits}')
print(f'Number of operands: {psum_num_ops}')

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
transform = alexnet_utils.use_alexnet_transform()
test_set = datasets.ImageNet(root="./data", split='val', transform=transform)
dataset_size = len(test_set)
print(f'Size of dataset: {dataset_size}')

start = int(sys.argv[3]) - 1
end = int(sys.argv[4])
data_indices = list(range(start, end))

print(f'Dataset index range: {data_indices[0]} - {data_indices[-1]}')

test_subset = Subset(test_set, data_indices)
test_loader = DataLoader(test_subset)


# inference
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_orig = False
run_qtz = False
run_custom = True

topk = (1, 5)
with torch.no_grad():
    if run_orig:
        _, fp32_top1, fp32_top5 = helper.evaluate_model_topk(
            model, test_loader,
            device=gpu_device,
            topk=topk,
            )
        print('Original model accuracy: ', end='')
        print(f'{fp32_top1:.5f} (top 1) / {fp32_top5:.5f} (top 5)')

    if run_qtz:
        _, qtz_top1, qtz_top5 = helper.evaluate_model_topk(
            qtz_model, test_loader,
            device=cpu_device,
            topk=topk,
            )
        print('Quantized model accuracy: ', end='')
        print(f'{qtz_top1:.5f} (top 1) / {qtz_top5:.5f} (top 5)')

    if run_custom:
        _, custom_top1, custom_top5 = helper.evaluate_model_topk(
            custom_model, test_loader,
            device=cpu_device,
            topk=topk,
            )
        print('Custom model accuracy: ', end='')
        print(f'{custom_top1:.5f} (top 1) / {custom_top5:.5f} (top 5)')

log_name = 'imagenet_acc.log'
with open(log_name, 'a') as log_file:
    log_file.write(f'Psum bits: {psum_bits:2}  /  ')
    log_file.write(f'Number of operands: {psum_num_ops:3}  /  ')
    log_file.write(f'Data range: {start+1} - {end}\n')
    log_file.write(f'Accuracy: ')
    log_file.write(f'{custom_top1:.5f} (top 1) / {custom_top5:.5f} (top 5)\n')
    log_file.write('-' * 75 + '\n')
    print(f'Record written in {log_name}')
