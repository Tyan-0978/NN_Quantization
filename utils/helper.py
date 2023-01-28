# ------------------------------------------------------------------------------
# Helper Functions for Quantization
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16, VGG16_Weights

import time 
import copy
import numpy as np
from tqdm.auto import tqdm

# --------------------------------------------------------------------
# train model
# --------------------------------------------------------------------
def train_model(model, train_loader, test_loader, num_epochs, lr=0.001, device=torch.device('cpu')):
  criterion = nn.CrossEntropyLoss()
  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

  for epoch in range(num_epochs):
    # Training
    model.train()

    running_loss = 0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader, desc='Training:'):

      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # statistics
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = running_corrects / len(train_loader.dataset)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(
      model=model, 
      test_loader=test_loader, 
      device=device, 
      criterion=criterion
    )

    print(
      "Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        epoch+1, train_loss, train_accuracy, eval_loss, eval_accuracy
      )
    )

  return

# --------------------------------------------------------------------
# evaluate model loss and accuracy
# --------------------------------------------------------------------
def evaluate_model(model, test_loader, device, criterion=None):
  model.eval()
  model.to(device)

  running_loss = 0
  running_corrects = 0

  for inputs, labels in tqdm(test_loader, desc='Evaluating:'):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    if criterion is not None:
      loss = criterion(outputs, labels).item()
    else:
      loss = 0

    # statistics
    running_loss += loss * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

  eval_loss = running_loss / len(test_loader.dataset)
  eval_accuracy = running_corrects / len(test_loader.dataset)

  return eval_loss, eval_accuracy

def evaluate_model_topk(model, test_loader, device, criterion=None):
  # TODO
  return

# --------------------------------------------------------------------
# model calibration for collecting quantization data statistics
# --------------------------------------------------------------------
def calibrate_model(model, loader):
  device = torch.device('cpu') # gpu is not supported for quantization
  model.to(device)
  model.eval()

  for inputs, _ in tqdm(loader, desc='Calibrating: '):
    inputs = inputs.to(device)
    #labels = labels.to(device)
    _ = model(inputs)

# --------------------------------------------------------------------
# inference latency measurement
# --------------------------------------------------------------------
def measure_inference_latency(
  model, 
  device,
  input_size=(1, 3, 64, 64),
  num_samples=100,
  num_warmups=10
):

  model.to(device)
  model.eval()

  x = torch.rand(size=input_size).to(device)

  with torch.no_grad():
    for _ in range(num_warmups):
      _ = model(x)
      # uncomment line below if cuda is available
      #torch.cuda.synchronize()

  with torch.no_grad():
    start_time = time.time()
    for _ in tqdm(range(num_samples)):
      _ = model(x)
      # uncomment line below if cuda is available
      #torch.cuda.synchronize()

    end_time = time.time()

  elapsed_time = end_time - start_time
  elapsed_time_ave = elapsed_time / num_samples

  return elapsed_time_ave

# --------------------------------------------------------------------
# save / load models
# --------------------------------------------------------------------
def save_model(model, model_filepath):
  torch.save(model, model_filepath)

def load_model(model_filepath, device=torch.device('cpu')):
  model = torch.load(model_filepath, map_location=device)
  return model

def save_torchscript_model(model, model_filepath):
  torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device=torch.device('cpu')):
  model = torch.jit.load(model_filepath, map_location=device)
  return model

# --------------------------------------------------------------------
# check equivalence of two models
# --------------------------------------------------------------------
def model_equivalence(
  model_1, model_2, 
  device=torch.device('cpu'), 
  rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,64,64)
):

  model_1.to(device)
  model_2.to(device)

  for _ in range(num_tests):
    x = torch.rand(size=input_size).to(device)
    y1 = model_1(x).detach().cpu().numpy()
    y2 = model_2(x).detach().cpu().numpy()
    if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
      print("Model equivalence test sample failed: ")
      print(y1)
      print(y2)
      return False

  return True

# --------------------------------------------------------------------
# random seeds
# --------------------------------------------------------------------
def set_random_seeds(random_seed=0):
  torch.manual_seed(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)
