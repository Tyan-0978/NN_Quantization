# ------------------------------------------------------------------------------
# Quantized Model Evaluation
# ------------------------------------------------------------------------------

import os
import torch

import helper

# settings -----------------------------------------------------------
# cuda is not supported for quantization
cpu_device = torch.device('cpu')
cuda_device = torch.device('cuda') if torch.cuda.is_available() else None
seed = 0
helper.set_random_seeds(seed)

print('Loading dataset ...')
_, test_loader = helper.prepare_cifar10_dataloader(
  num_workers=0, train_batch_size=128, eval_batch_size=256
)
print('Done')
print('')

# load models --------------------------------------------------------
model_dir = './models/'
fp32_model_filename = 'alexnet_cifar10.pt'
fp32_model_filepath = os.path.join(model_dir, fp32_model_filename) 
qtz_model_filename = 'alexnet_qtz_act_histogram.pt'
qtz_model_filepath = os.path.join(model_dir, qtz_model_filename) 

print(f'Loading FP32 model from {fp32_model_filepath} ...')
fp32_model = helper.load_model(fp32_model_filepath)
fp32_model.eval()
print('Done')

print(f'Loading quantized model from {qtz_model_filepath} ...')
qtz_model = helper.load_torchscript_model(qtz_model_filepath)
qtz_model.eval()
print('Done')
print('')

# test accuracy ------------------------------------------------------
print('Evaluating quantized model ...')
_, qtz_eval_acc = helper.evaluate_model(
  model=qtz_model, 
  test_loader=test_loader, 
  device=cpu_device, 
  criterion=None
)
print('Done')

print('Evaluating FP32 model ...')
_, fp32_eval_acc = helper.evaluate_model(
  model=fp32_model, 
  test_loader=test_loader, 
  device=cpu_device, 
  criterion=None
)
print('Done')

print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_acc))
print("INT8 evaluation accuracy: {:.3f}".format(qtz_eval_acc))

# measure latency ----------------------------------------------------
print('Measuring inference latency ...')
fp32_cpu_inference_latency = helper.measure_inference_latency(
  model=fp32_model, 
  device=cpu_device, 
  input_size=(1,3,64,64), 
  num_samples=50
)
print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))

int8_cpu_inference_latency = helper.measure_inference_latency(
  model=qtz_model, 
  device=cpu_device, 
  input_size=(1,3,64,64), 
  num_samples=50
)
print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))

'''
if cuda_device:
  fp32_gpu_inference_latency = helper.measure_inference_latency(
    model=fp32_model, 
    device=cuda_device, 
    input_size=(1,3,64,64), 
    num_samples=50
  )
  print("FP32 GPU Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency*1000))

  int8_gpu_inference_latency = helper.measure_inference_latency(
    model=qtz_model, 
    device=cuda_device, 
    input_size=(1,3,64,64), 
    num_samples=50
  )
  print("INT8 GPU Inference Latency: {:.2f} ms / sample".format(int8_gpu_inference_latency*1000))
'''

print('')
