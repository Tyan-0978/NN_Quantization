# ------------------------------------------------------------------------------
# VGG16 Training using CIFAR10 Dataset
# ------------------------------------------------------------------------------

import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models import VGG16_Weights
from tqdm.auto import tqdm

from utils import helper, datasets, vgg16_utils

random_seed = 0
helper.set_random_seeds(random_seed=random_seed)

# prepare CIFAR10 dataset --------------------------------------------
print('Preparing CIFAR10 dataset ...')

train_batch_size = 10
eval_batch_size = 1
transform = vgg16_utils.use_vgg16_transform()

train_loader, test_loader = datasets.prepare_cifar10_loaders(
  data_transform=transform,
  train_batch_size=train_batch_size, 
  eval_batch_size=eval_batch_size
)

print('Done\n')

# create pretrained VGG16 model --------------------------------------
print('Creating VGG16 model for CIFAR10 ...')

model = vgg16_utils.create_vgg16_model()

# last layer output size is modified for CIFAR10 (10 classes)
model.classifier[3] = torch.nn.Linear(4096,1024)
model.classifier[6] = torch.nn.Linear(1024,10)
print(model)

print('Done\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')

# train and evaluate VGG16 with CIFAR10 -----------------------------------------
print('Start training ...')

num_epochs = 1
learning_rate = 0.001

helper.train_model(
  model=model, 
  train_loader=train_loader, 
  test_loader=test_loader, 
  num_epochs=num_epochs,
  lr=learning_rate,
  device=device
)

print('Finish training.\n')

print('Start evaluation ...')

_, eval_accuracy = helper.evaluate_model_topk(
  model=model, 
  test_loader=test_loader, 
  device=device,
)

print(f'Model accuracy: {eval_accuracy}')
print('')

'''
# test for all classes
print('Testing for all classes ...')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = [0.] * 10
class_total = [0.] * 10

with torch.no_grad():
  for data in tqdm(test_loader):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1

for i in range(10):
  print('Accuracy of %5s : %2d %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]
  ))

avg = 0
for i in range(10):
  temp = (100 * class_correct[i] / class_total[i])
  avg = avg + temp
avg = avg / 10

print(f'Average accuracy = {avg}\n')
'''

# save model ---------------------------------------------------------
save_model = False
model_name = f'vgg16_cifar10_{int(eval_accuracy)}.pt'

model_dir = './models/'
model_path = os.path.join(model_dir, model_name)
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

if save_model:
  helper.save_model(model, model_path)
  print(f'Model saved at {model_path}')
  print('')
