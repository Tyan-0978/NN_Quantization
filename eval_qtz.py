# ------------------------------------------------------------------------------
# Quantized Model Evaluation
# ------------------------------------------------------------------------------

import os
import torch
from utils import helper, datasets, alexnet_utils, vgg16_utils

# settings -----------------------------------------------------------
# cuda is not supported for quantization
qtz_device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 0
helper.set_random_seeds(seed)

print('Loading dataset ...')

transform = alexnet_utils.use_alexnet_transform()
#transform = vgg16_utils.use_vgg16_transform()

_, test_loader = datasets.prepare_cifar10_loaders(transform)

print('Done')
print('')

# load models --------------------------------------------------------
fp32_model_filename = 'alexnet_cifar10.pt'
qtz_model_filename = 'alexnet_cifar10_qtz_histogram.pt'

model_dir = './models/'
fp32_model_filepath = os.path.join(model_dir, fp32_model_filename) 
qtz_model_filepath = os.path.join(model_dir, qtz_model_filename) 

print(f'Loading FP32 model from {fp32_model_filepath} ...', end=' ')

fp32_model = helper.load_model(fp32_model_filepath)
fp32_model.eval()

print('Done')

print(f'Loading quantized model from {qtz_model_filepath} ...', end=' ')

qtz_model = helper.load_torchscript_model(qtz_model_filepath)
qtz_model.eval()

print('Done')
print('')

# test accuracy ------------------------------------------------------
print('Evaluating quantized model ...')

topk = (1, )

_, qtz_eval_acc = helper.evaluate_model_topk(
  model=qtz_model, 
  test_loader=test_loader, 
  device=qtz_device, 
  topk=topk
)
print('Done')

print('Evaluating FP32 model ...')
_, fp32_eval_acc = helper.evaluate_model_topk(
  model=fp32_model, 
  test_loader=test_loader, 
  device=device, 
  topk=topk
)
print('Done')

print("FP32 evaluation accuracy: {:.5f}".format(fp32_eval_acc))
print("INT8 evaluation accuracy: {:.5f}".format(qtz_eval_acc))

# measure latency ----------------------------------------------------
print('Measuring inference latency ...')

fp32_inf_latency = helper.measure_inference_latency(
  model=fp32_model, 
  device=device, 
  input_size=(10,3,256,256), 
  num_samples=50
)

print(f'FP32 CPU inference latency: {(fp32_inf_latency * 1000):.3f} ms / sample')

int8_inf_latency = helper.measure_inference_latency(
  model=qtz_model, 
  device=qtz_device, 
  input_size=(1,3,64,64), 
  num_samples=50
)
print(f'INT8 CPU inference latency: {(int8_inf_latency * 1000):.3f} ms / sample')
