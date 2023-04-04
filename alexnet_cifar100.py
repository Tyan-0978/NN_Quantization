# ------------------------------------------------------------------------------
# AlexNet Training using CIFAR100 Dataset
# ------------------------------------------------------------------------------

import os
import torch
import torchvision

from utils import helper, datasets, alexnet_utils

print('=== AlexNet training on CIFAR100 ===')

random_seed = 0
helper.set_random_seeds(random_seed=random_seed)

# prepare CIFAR100 dataset --------------------------------------------
print('Preparing CIFAR100 dataset ...')

train_batch_size = 100
eval_batch_size = 1
transform = alexnet_utils.use_alexnet_transform()

train_loader, test_loader = datasets.prepare_cifar100_loaders(
    data_transform=transform,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size
    )

print('Done')
print('')

# create pretrained AlexNet model ----------------------------------------------
print('Creating AlexNet model for CIFAR100 ...')

model = alexnet_utils.create_alexnet_model()

# modify FC layers for CIFAR100 (output 100 classes)
model.classifier[4] = torch.nn.Linear(4096,1024)
model.classifier[6] = torch.nn.Linear(1024,100)
model.eval()
print(model)

print('Done')
print('')

# use GPU for faster training (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# train and evaluate AlexNet with CIFAR100 ----------------------------------------
print('Start training (this will take a while) ...')

num_epochs = 1

helper.train_model(
  model=model, 
  train_loader=train_loader, test_loader=test_loader, 
  num_epochs=num_epochs, device=device
)

print('Finish training.')
print('')

print('Start evaluation ...')

topk = (1, 5)
with torch.no_grad():
    _, top1_acc, top5_acc = helper.evaluate_model_topk(
        model, test_loader, device, topk=topk
        )

print(f'Top 1 accuracy: {top1_acc:.5f}')
print(f'Top 5 accuracy: {top5_acc:.5f}')
print('')

# save model ---------------------------------------------------------
save_model = False

if save_model:
    model_name = f'alexnet_cifar100_acc{int(top1_acc * 100)}.pt'

    model_path = os.path.join(model_dir, model_name)
    model_dir = './models/'
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    helper.save_model(model, model_path)
    print(f'Model saved at {model_path}')
    print('')

# end
