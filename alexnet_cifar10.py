# ------------------------------------------------------------------------------
# AlexNet Training using CIFAR10 Dataset
# ------------------------------------------------------------------------------

import os
import torch
import torchvision
from tqdm.auto import tqdm

from utils import helper, datasets, alexnet_utils

print('=== Start AlexNet training on CIFAR10 ===')

random_seed = 0
helper.set_random_seeds(random_seed=random_seed)

# prepare CIFAR10 dataset --------------------------------------------
print('Preparing CIFAR10 dataset ...')

train_batch_size = 4
eval_batch_size = 1
transform = alexnet_utils.use_alexnet_transform()

train_loader, test_loader = datasets.prepare_cifar10_loaders(
  data_transform=transform,
  train_batch_size=train_batch_size, 
  eval_batch_size=eval_batch_size
)

print('Done')
print('')


# create pretrained AlexNet model ----------------------------------------------
print('Creating AlexNet model for CIFAR10 ...')

model = alexnet_utils.create_alexnet_model()

# modify FC layers for CIFAR10 (output 10 classes)
model.classifier[4] = torch.nn.Linear(4096,1024)
model.classifier[6] = torch.nn.Linear(1024,10)
model.eval()
print(model)

print('Done')
print('')

# use GPU for faster training (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')

# train and evaluate AlexNet with CIFAR10 -----------------------------------------
print('Start training (this will take a while) ...')

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

print('Done training.')
print('')

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
  for data in tqdm(test_loader, desc='Testing: '):
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
model_dir = './models/'
model_name = f'alexnet_cifar10_{int(eval_accuracy * 100)}.pt'
model_path = os.path.join(model_dir, model_name)
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

if save_model:
  helper.save_model(model, model_path)
  print(f'Model saved at {model_path}')
  print('')
