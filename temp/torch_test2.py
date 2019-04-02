#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) #, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False) #, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, [1,2,0]))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(net, testloader):
    with torch.no_grad():
        all_corrects = all_data = 0
        for inputs, labels in testloader:
            y = net(inputs)
            _, predicts = torch.max(y, 1)
            corrects = (predicts == labels).sum().item()
            all_corrects += corrects
            all_data += len(labels)
    return all_corrects/all_data

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(trainloader, 1):
        optimizer.zero_grad()
        y = net(images)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if i % 2000 == 0:
            acc = accuracy(net, testloader)
            print('epoch %d - %d: loss %f, test accuracy %f' % (epoch, i, epoch_loss/2000, acc))
            epoch_loss = 0.0




