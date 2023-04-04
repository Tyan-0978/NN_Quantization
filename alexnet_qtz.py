# ------------------------------------------------------------------------------
# AlexNet Quantization
# ------------------------------------------------------------------------------

import os
import copy
import pickle

import torch
from torch.quantization import QConfig, MinMaxObserver, HistogramObserver
import torchvision

from utils import helper, datasets, alexnet_utils
from utils.QuantizedModel import QuantizedModel

# settings -----------------------------------------------------------
random_seed = 0
helper.set_random_seeds(random_seed=random_seed)

device = torch.device("cpu") # CUDA is not supported for quantization

print('Loading dataset ...')

transform = alexnet_utils.use_alexnet_transform()
train_batch_size = 1
eval_batch_size = 1

train_loader, test_loader = datasets.prepare_imagenet_loaders(
    data_transform=transform,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
)

print('Done')
print('')

# load Alexnet model
model_dir = './models/'
fp32_model_name = 'alexnet_cifar10.pt'
fp32_model_path = os.path.join(model_dir, fp32_model_name)
#print(f'Loading FP32 model from {fp32_model_path} ...')

fp32_model = helper.load_model(fp32_model_path)
#fp32_model = alexnet_utils.create_alexnet_model()

fp32_model.to(device)
print(fp32_model)

print('Done')
print('')

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

# save quantized model and quantization parameters
save_model = False

if save_model:
    prefix = f'alexnet_cifar10_qtz_acc{int(eval_accuracy*100)}'

    params_dir = 'params/'
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    params_name = prefix + '_params.pkl'
    params_path = os.path.join(params_dir, params_name)
    quantized_model.save_params(params_path)
    print(f'Quantization parameters saved to {params_path}')

    qtz_model_name = prefix + '_model.pt'
    qtz_model_path = os.path.join(model_dir, qtz_model_name)
    helper.save_torchscript_model(quantized_model, qtz_model_path)
    print(f'Quantization model saved to {qtz_model_path}')

# end
