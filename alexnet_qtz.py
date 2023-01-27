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

# transform for AlexNet
mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean, std_dev)
])

train_loader, test_loader = helper.prepare_cifar10_dataloader(data_transform=transform)

print('Done')
print('')

# load Alexnet model (pretrained by CIFAR10)
# pretrained model should be placed at the path printed above
print(f'Loading FP32 model from {fp32_model_filepath} ...')
fp32_model = helper.load_model(fp32_model_filepath)
fp32_model.to(cpu_device)
print('Done')
print('')

'''
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
'''

# quantization -------------------------------------------------------
print('Starting quantization ...')
quantized_model = QuantizedModel(model_fp32=fp32_model)

'''
default_quantization_config = torch.quantization.default_qconfig
quantized_model.qconfig = default_quantization_config
'''
act_config = torch.quantization.HistogramObserver.with_args(
  dtype=torch.quint8,
  qscheme=torch.per_tensor_symmetric,
)
weight_config = torch.quantization.MinMaxObserver.with_args(
  dtype=torch.qint8,
  qscheme=torch.per_tensor_symmetric,
)

quantization_config = torch.quantization.QConfig(
  activation=act_config,
  weight=weight_config
)
quantized_model.qconfig = quantization_config

print('Quantization configurations:')
print(quantized_model.qconfig)
print('')

torch.quantization.prepare(quantized_model, inplace=True)

# Use training data for calibration.
print('Calibrating quantized model ...')
helper.calibrate_model(model=quantized_model, loader=train_loader)
print('Calibration finished')

quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model.eval()
print('Quantization finished')
print('')

print('Start evaluation ...')

_, eval_accuracy = helper.evaluate_model(
  model=quantized_model,
  test_loader=test_loader,
  device=cpu_device,
)

print(f'Model accuracy: {eval_accuracy}')
print('')

# save quantized model
res = input('Save model? (y/n): ')
while res not in ['', 'y', 'n']:
  res = input('Unknown response. Save model? (y/n): ')
if res == 'n':
  print('Model not saved. Program is ended.')
else:
  qtz_model_filename = input('Insert file name: ')
  if not qtz_model_filename: os.exit
  qtz_model_filepath = os.path.join(model_dir, qtz_model_filename)
  #helper.save_model(quantized_model, qtz_model_filepath)
  helper.save_torchscript_model(quantized_model, qtz_model_filepath)
  print(f'Quantization model saved to {qtz_model_filepath}')

print('')
