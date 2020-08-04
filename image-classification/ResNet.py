# --------------------------------------------------------------------------- #
# 
# Copyright 2020, Southeast University, Liu Pengxiang
# 
# It is the source code for the final project of course "Neural Network and 
# Machine Learning". We adopt ResNet model to perform image classification on
# the VOC-2012 data set.
#
# --------------------------------------------------------------------------- #


import os
import json
import time
import torch
import random
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from lxml import etree
from torchvision import transforms, datasets
from torch.utils.data import Dataset


class BasicBlock(nn.Module):
    exp = 1

    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channel, out_channels = out_channel,
            kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels = out_channel, out_channels = out_channel,
            kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    exp = 4

    def __init__(self, in_channel, out_channel, stride = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channel, out_channels = out_channel,
            kernel_size = 1, stride = 1, bias = False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels = out_channel, out_channels = out_channel,
            kernel_size = 3, stride = stride, bias = False, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels = out_channel, out_channels = out_channel * self.exp,
            kernel_size = 1, stride = 1, bias = False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.exp)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes = 1000, include_top = True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size = 7, stride = 2,
                               padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64 , blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride = 2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.exp, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode = 'fan_out', nonlinearity = 'relu'
                )

    def _make_layer(self, block, channel, block_num, stride = 1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.exp:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel, channel * block.exp, kernel_size=1, 
                    stride = stride, bias = False),
                nn.BatchNorm2d(channel * block.exp))

        layers = []
        layers.append(block(
            self.in_channel, channel, downsample=downsample, stride = stride
        ))
        self.in_channel = channel * block.exp

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes = 1000, include_top = True):
    return ResNet(BasicBlock, [3, 4, 6, 3], 
                  num_classes = num_classes, include_top = include_top)

def resnet101(num_classes = 1000, include_top = True):
    return ResNet(Bottleneck, [3, 4, 23, 3], 
                  num_classes = num_classes, include_top = include_top)

def build_resnet(layers, num_classes = 1000, include_top = True):
    layer_dict = {}
    layer_dict['18' ] = [2, 2,  2, 2]
    layer_dict['34' ] = [3, 4,  6, 3]
    layer_dict['50' ] = [3, 4,  6, 3]
    layer_dict['101'] = [3, 4, 23, 3]
    layer_dict['152'] = [3, 8, 36, 3]
    try:
        if layers == '18' or layers == '34':
            block = BasicBlock
        else:
            block = Bottleneck
        return ResNet(block, layer_dict[layers], 
                      num_classes = num_classes, include_top = include_top)
    except Exception as exception:
        print(exception)
        exit(-1)

# main function for image classification
def image_classification():

    # root dir
    image_path = 'D:\\Data Set\\image_dataset\\VOCdevkit\\VOC_image_cls'
    trans_path = 'D:\\Data Set\\transfer_path'

    # determine the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # transform
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # load train data set
    batch_size = 64
    train_dataset = datasets.ImageFolder(
        root = os.path.join(image_path, "train"), 
        transform = data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, 
        num_workers = 0)

    # load validation data set
    batch_size = 64
    valid_dataset = datasets.ImageFolder(
        root = os.path.join(image_path, "valid"), 
        transform = data_transform["valid"])
    valid_num = len(valid_dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size = batch_size, shuffle = False,
        num_workers = 0)
    
    # modeling
    layers = '34'
    net = build_resnet(layers, 20)

    # transfer learning
    # weight_path = os.path.join(trans_path, "resnet{}-pre.pth".format(layers))
    # missing_keys, unexpected_keys = net.load_state_dict(
    #     torch.load(weight_path), strict = False)
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 20)

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    best_acc = 0.0
    save_path = os.path.join(trans_path, "resNet{}-train.pth".format(layers))
    logging = {'trace': []}
    for epoch in range(100):

        # train
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start = 0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "=" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rprogress: [{}->{}] {:^3.0f}% train loss:{:.4f}".format(
                a, b, int(rate * 100), loss), end = "")
        print()
        t2 = time.perf_counter()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim = 1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / valid_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f test_accuracy: %.3f time: %.2fs'
                  % (epoch + 1, running_loss / step, val_accurate, t2 - t1))
        
        # log
        logging['trace'].append([epoch + 1, running_loss / step, val_accurate])
    
    path = os.path.join(os.getcwd(), 'log.mat')
    sio.savemat(path, logging)


if __name__ == "__main__":
    image_classification()