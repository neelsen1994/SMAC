from __future__ import print_function

from datasets import *
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import gzip
import json
import os
import pickle
import numpy as np
import time
import logging

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

os.environ["CUDA_VISIBLE_DEVICES"]="0"

batch_size = 100

DATA_PATH = './dataset'
#MODEL_STORE_PATH = 'C:\\Users\Andy\PycharmProjects\pytorch_models\\'


# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = KMNIST(data_dir=DATA_PATH, train=True, transform=trans)
test_dataset = KMNIST(data_dir=DATA_PATH, train=False, transform=trans)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#X,y = torch.load('C:\\Users\\NEIL\\OneDrive\\Documents\\Study\\Masters Project\\CNN KMNIST\\data\\KMNIST\\processed\\training.pt')

#print("Train", X.shape)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)



    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def cnn_from_cfg(cfg):
    model = ConvNet()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(cfg["epo"]):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
    
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, cfg["epo"], i + 1, total_step, loss.item(),
                              (correct / total) * 100))
    
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        accScore = (correct / total) 
        error = 1 - accScore
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(accScore * 100))
    
    return error    


logging.basicConfig(level=logging.DEBUG)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# There are some hyperparameters shared by all kernels
lr = UniformFloatHyperparameter("lr", 0.0001, 0.1, default_value=0.1)
epo = UniformIntegerHyperparameter("epo", 1, 10, default_value=5) 
cs.add_hyperparameters([lr, epo])

scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 150,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })


# It returns: Status, Cost, Runtime, Additional Infos
#def_value = cnn_from_cfg(cs.get_default_configuration())
#print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimization started...")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=cnn_from_cfg)

incumbent = smac.optimize()

inc_value = cnn_from_cfg(incumbent)

print("Optimized Value: %.4f" % (inc_value))

ac = (1 - inc_value) * 100

print("Best Configuration: %.2f" % (ac))
