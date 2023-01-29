# ------------------------------------------------------------------------------
# Quantization for VGG16 on CIFAR10
# ------------------------------------------------------------------------------

import os
import copy
import torch
from torch.quantization import QConfig, MinMaxObserver, HistogramObserver
import torchvision
from utils import helper, datasets, vgg16_utils
from utils.QuantizedModel import QuantizedModel

# settings -----------------------------------------------------------
random_seed = 0
helper.set_random_seeds(random_seed=random_seed)

device = torch.device('cpu') # CUDA is not supported for quantization

print('Loading dataset ...')

transform = vgg16_utils.use_vgg16_transform()
train_batch_size = 1
eval_batch_size = 1

train_loader, test_loader = datasets.prepare_cifar10_loaders(
  data_transform=transform,
  train_batch_size=train_batch_size,
  eval_batch_size=eval_batch_size
)

print('Done')
print('')

# load VGG16 model (pretrained by CIFAR10)
model_dir = './models/'
fp32_model_name = 'vgg16_cifar10.pt'
fp32_model_path = os.path.join(model_dir, fp32_model_name)

print(f'Loading FP32 model from {fp32_model_path} ...')

fp32_model = helper.load_model(fp32_model_path)
fp32_model.to(device)
print(fp32_model)

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
print('Start quantization ...')

quantized_model = QuantizedModel(model_fp32=fp32_model)

'''
default_quantization_config = torch.quantization.default_qconfig
quantized_model.qconfig = default_quantization_config
'''
act_config = HistogramObserver.with_args(
  dtype=torch.quint8,
  qscheme=torch.per_tensor_symmetric,
)
weight_config = HistogramObserver.with_args(
  dtype=torch.qint8,
  qscheme=torch.per_tensor_symmetric,
)

quantization_config = QConfig(activation=act_config, weight=weight_config)
quantized_model.qconfig = quantization_config

print('Quantization configurations:')
print(quantized_model.qconfig, '\n')

torch.quantization.prepare(quantized_model, inplace=True)

# Use training data for calibration.
print('Calibrating quantized model ...')

helper.calibrate_model(model=quantized_model, loader=train_loader)

print('Calibration finished')

quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model.eval()

print('Quantization finished\n')

print('Start evaluation ...')

_, eval_accuracy = helper.evaluate_model_topk(
  model=quantized_model,
  test_loader=test_loader,
  device=device,
)

print(f'Model accuracy: {eval_accuracy}\n')

# save quantized model
save_model = True
qtz_model_name = 'vgg16_cifar10_qtz_histogram.pt'

if save_model:
  qtz_model_path = os.path.join(model_dir, qtz_model_name)
  helper.save_torchscript_model(quantized_model, qtz_model_path)
  print(f'Quantization model saved to {qtz_model_path}')
