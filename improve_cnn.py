#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import time
# hyperparamater setting
train_path='./archive/Train'
test_path='./archive/Test1'
batch_size=100
learn_rate=0.001
momentum=0.9
class_num=43
num_train=10000
classes=classes = range(0,class_num)
# data processing
# first （0.5，0.5，0.5） is mean for R G B, second (0.5, 0.5, 0.5) is standard deviation for RGB
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((128,128)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root=test_path,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(256 * 16 * 16, 160)
        self.fc2 = nn.Linear(160, 100)
        self.fc3 = nn.Linear(100, 43)

    def forward(self, out):
        # input dim: 3*128*128
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        # dim: 64*128*128
        out = self.pool(out)
        out = self.drop(out)
        # dim: 64*64*64
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        # dim: 128*64*64
        out = self.pool(out)
        out = self.drop(out)
        # dim: 128*32*32
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        # dim: 256*32*32
        out = self.pool(out)
        out = self.drop(out)
        # dim: 256*16*16
        # change the dim for fully connection (batch_size, 16, 5, 5) -> (batch_size, 16*5*5)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        # dim: (batch_size, 256*16*16)
        # 2-layer Fully connection
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.zeros_(m.bias)
net = Net()
net.apply(weights_init)
print(net)
if (device=="cuda:0"):
    net.to(device)


# CrossEntropyLoss() include softmax step
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learn_rate)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_arr=[]
train_accuracy_arr=[]
test_accuracy_arr=[]
iter=0
# training process
for epoch in range(2):  # loop over the dataset multiple times
# i is index, data[0] is data, data[1] is label
    for i, data in enumerate(trainloader, 0):
        start = time.time()
        # get the inputs; data is a list of [inputs, labels] and convert the data to GPU
        # each Iteration deal with number of batch size images
        if (device=="cuda:0"):
            inputs, labels = data[0].to(device), data[1].to(device)
        else:
            inputs, labels = data[0], data[1]

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        outputs = net(inputs)
        # calculate train Accuracy in each step
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_accuracy_arr.append(100 * correct / total)
        loss_arr.append(loss.data.item())
        end = time.time()
        runtime = end - start
        print('batch: %d, train loss: %.4f, train acc: %.2f, time: %.2f' %
              (i, loss.data.item(), train_accuracy_arr[i], runtime))
        iter=iter+1

x=list(range(0, iter))
plt.plot(x, loss_arr, color='red', linewidth=2.0)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.show()
print('Finished Training')


# calculate Testing Accuracy
np.set_printoptions(threshold=np.inf)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        if (device=="cuda:0"):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data[0], data[1]
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("-------------------------------------------------------------------------")

print("correct: ", correct)
print("total: ", total)
print('Testing Accuracy of the network on the 12000+ test images: %d %%' % (
    100 * correct / total))

# calculate test Accuracy for each class and produce confusion_matrix
class_correct = list(0. for i in range(43))
class_total = list(0. for i in range(43))
cfm=np.zeros((43,43))
with torch.no_grad():
    for data in testloader:
        if (device=="cuda:0"):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data[0], data[1]
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        i=0
        for n in images:
            label = labels[i]
            cfm[predicted[i]][label] += 1
            class_correct[label] += c[i].item()
            class_total[label] += 1
            i=i+1

print(cfm)
# draw bar chat for the Accuracy of each class
acc_test_each=[]
for i in range(43):
    print('Testing Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    acc_test_each.append(class_correct[i] / class_total[i])

plt.figure(figsize=(15,4))
plt.bar(range(len(classes)), acc_test_each,color='r',tick_label=classes)
plt.tight_layout()
plt.show()

# calculate training Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        if (device=="cuda:0"):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data[0], data[1]
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("-------------------------------------------------------------------------")
print('training Accuracy of the network on the 30000+ train images: %d %%' % (
    100 * correct / total))

# calculate training Accuracy for each class and produce confusion_matrix
cfm=np.zeros((43,43))
class_correct = list(0. for i in range(43))
class_total = list(0. for i in range(43))
with torch.no_grad():
    for data in trainloader:
        if (device=="cuda:0"):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data[0], data[1]
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        i=0
        for x in images:
            label = labels[i]
            cfm[predicted[i]][label] += 1
            class_correct[label] += c[i].item()
            class_total[label] += 1
            i=i+1


print(cfm)
acc_train_each=[]
for i in range(43):
    print('Training Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    acc_train_each.append(class_correct[i] / class_total[i])
# draw bar chat for the Accuracy of each class
plt.figure(figsize=(15,4))
plt.bar(range(len(classes)), acc_train_each,color='r',tick_label=classes)
plt.tight_layout()
plt.show()
