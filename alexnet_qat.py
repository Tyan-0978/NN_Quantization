# ------------------------------------------------------------------------------
# AlexNet Quantization-Aware Training 
# ------------------------------------------------------------------------------

import os
import torch
import torchvision

from utils import helper, datasets, alexnet_utils
from utils.QuantizedModel import QuantizedModel as QtzModel

print('=== AlexNet Quantization-Aware Training ===')

random_seed = 0
helper.set_random_seeds(random_seed=random_seed)


print('Preparing dataset ...')

train_batch_size = 100
eval_batch_size = 1
transform = alexnet_utils.use_alexnet_transform()

train_loader, test_loader = datasets.prepare_cifar100_loaders(
  data_transform=transform,
  train_batch_size=train_batch_size,
  eval_batch_size=eval_batch_size
)

print('Finished\n')


print('Creating AlexNet model ...')

model_name = 'alexnet_cifar100_acc67.pt'

model_dir = './models/'
model_path = os.path.join(model_dir, model_name)

#model = alexnet_utils.create_alexnet_model()
model = helper.load_model(model_path)

# modify FC layers for CIFAR100 (output 100 classes)
#model.classifier[4] = torch.nn.Linear(4096,1024)
#model.classifier[6] = torch.nn.Linear(1024,100)
print(model)

print('Finished\n')


# QAT configuration and preparing
qat_model = QtzModel(model_fp32=model)

qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(qat_model, inplace=True)


# training
# GPU does not support quantization
device = torch.device('cpu')
print(f'Current device: {device}')

print('Start training (this will take a while) ...')

num_epochs = 2

helper.train_model(model=qat_model, 
                   train_loader=train_loader, 
		   test_loader=test_loader, 
		   num_epochs=num_epochs, 
		   device=device)

qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

num_epochs = 1

helper.train_model(model=qat_model, 
                   train_loader=train_loader, 
		   test_loader=test_loader, 
		   num_epochs=num_epochs, 
		   device=device)

qat_model.apply(torch.ao.quantization.disable_observer)

num_epochs = 7

helper.train_model(model=qat_model, 
                   train_loader=train_loader, 
		   test_loader=test_loader, 
		   num_epochs=num_epochs, 
		   device=device)

print('Finish training.')


quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)

print('Model converted to quantized model.\n')

print('Start evaluation ...')

topk = (1, 5)
with torch.no_grad():
    _, top1_acc, top5_acc = helper.evaluate_model_topk(
        quantized_model, test_loader, device, topk=topk
        )

print(f'Top 1 accuracy: {top1_acc:.5f}')
print(f'Top 5 accuracy: {top5_acc:.5f}')
print('')


save_model = False

if save_model:
    model_name = f'alexnet_qat_cifar100_acc{int(top1_acc * 100)}.pt'

    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    helper.save_torchscript_model(quantized_model, model_path)
    print(f'Model saved at {model_path}\n')
