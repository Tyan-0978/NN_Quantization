# ------------------------------------------------------------------------------
# Quantization on AlexNet for CIFAR10
# ------------------------------------------------------------------------------

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from tqdm.auto import tqdm

import helper
from QuantizedModel import QuantizedModel

# settings -----------------------------------------------------------
random_seed = 0
num_classes = 10
cpu_device = torch.device("cpu")
# cuda is not supported for quantization

model_dir = './models/'
fp32_model_filename = 'alexnet_cifar10.pt'
fp32_model_filepath = os.path.join(model_dir, fp32_model_filename)

helper.set_random_seeds(random_seed=random_seed)

print('Loading dataset ...')
train_loader, test_loader = helper.prepare_cifar10_dataloader(
  num_workers=8, train_batch_size=128, eval_batch_size=256
)
print('Done')
print('')

# load Alexnet model (pretrained by CIFAR10)
# pretrained model should be placed at the path printed above
print(f'Loading FP32 model from {fp32_model_filepath} ...')
fp32_model = helper.load_model(fp32_model_filepath)
fp32_model.to(cpu_device)
print('Done')
print('')

# model fusion -------------------------------------------------------
print('Fusing FP32 Model ...')
fused_model = copy.deepcopy(fp32_model)

fp32_model.eval()
fused_model.eval()

for module_name, module in fused_model.named_children():
  if "layer" in module_name:
    for basic_block_name, basic_block in module.named_children():
      torch.quantization.fuse_modules(
        basic_block, 
	[["conv1", "bn1", "relu1"], ["conv2", "bn2"]], 
	inplace=True
      )
      for sub_block_name, sub_block in basic_block.named_children():
        if sub_block_name == "downsample":
          torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

print('Done')
assert helper.model_equivalence(
  model_1=fp32_model, 
  model_2=fused_model, 
  device=cpu_device, 
  rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,64,64)
), "Fused model is not equivalent to the original model!"
print('Model equivalence test passed')
print('')

# quantization -------------------------------------------------------
print('Starting quantization ...')
quantized_model = QuantizedModel(model_fp32=fused_model)

# Select quantization schemes from 
# https://pytorch.org/docs/stable/quantization-support.html
#quantization_config = torch.quantization.get_default_qconfig("fbgemm")
# Custom quantization configurations
quantization_config = torch.quantization.default_qconfig
#quantization_config = torch.quantization.QConfig(
#  activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
#  weight=torch.quantization.MinMaxObserver.with_args(
#    dtype=torch.qint8, 
#    qscheme=torch.per_tensor_symmetric
#  )
#)
quantized_model.qconfig = quantization_config

print('Quantization configurations:')
print(quantized_model.qconfig)

torch.quantization.prepare(quantized_model, inplace=True)

# Use training data for calibration.
print('Calibrating quantized model ...')
helper.calibrate_model(model=quantized_model, loader=train_loader)
print('Calibration finished')

quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model.eval()
print('Quantization finished')

# save quantized model
qtz_model_filename = 'alexnet_cifar10_qtz_default_config.pt'
qtz_model_filepath = os.path.join(model_dir, qtz_model_filename)
helper.save_model(quantized_model, qtz_model_filepath)
#helper.save_torchscript_model(quantized_model, quantized_model_filepath)
print(f'Quantization model saved to {qtz_model_filepath}')
print('')
