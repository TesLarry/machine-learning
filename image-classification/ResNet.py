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
import torch
import random
import torch.nn as nn
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


# Read the data set of PASCAL VOC2012
class VOC2012DataSet(Dataset):

    def __init__(self, voc_root, transforms, train_mode = True, initial = 0):

        # direction
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or valid.txt file
        if initial != 0:
            self.data_split(0.3)
        if train_mode == 1:
            txt_list = os.path.join(os.getcwd(), "train.txt")
        else:
            txt_list = os.path.join(os.getcwd(), "valid.txt")
        n = 1
        with open(txt_list) as read:
            self.xml_list = [
                os.path.join(self.annotations_root, line.strip() + ".xml")
                for line in read.readlines()
            ]

        # read class_indict
        try:
            json_file = open('./pascal_voc_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def data_split(self, rate):
        # split the data as train set and validation set
        valid_rate = rate
        files_path = self.annotations_root
        files_name = sorted(
            [file.split(".")[0] for file in os.listdir(files_path)]
        )
        valid_index = random.sample(
            range(0, len(files_name)), k = int(len(files_name) * valid_rate)
        )
        # create training and validation set
        train_files = []
        valid_files = []
        for index, file_name in enumerate(files_name):
            if index in valid_index:
                valid_files.append(file_name)
            else:
                train_files.append(file_name)
        # save
        try:
            train_f = open("train.txt", "x")
            valid_f = open("valid.txt", "x")
            train_f.write("\n".join(train_files))
            valid_f.write("\n".join(valid_files))
        except FileExistsError as e:
            print(e)
            exit(1)

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):

        # convert xml to dict (recursive_parse_xml_to_dict)
        if len(xml) == 0:  # root level
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


# main function for image classification
def image_classification():

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

    # root dir
    voc_root = 'd:\Data Set\image_dataset'

    # load train data set
    train_data_set = VOC2012DataSet(voc_root, data_transform["train"], 1)
    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size = 16, shuffle = 1, num_workers = 0)
    train_num = len(train_data_set)

    # load validation data set
    valid_data_set = VOC2012DataSet(voc_root, data_transform["valid"], 0)
    valid_loader = torch.utils.data.DataLoader(
        valid_data_set, batch_size = 16, shuffle = 0, num_workers = 0)
    valid_num = len(valid_data_set)
    
    net = resnet34()
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 20)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    save_path = './resNet34.pth'
    for epoch in range(3):
        # train
        net.train()
        running_loss = 0.0
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
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        print()

if __name__ == "__main__":
    image_classification()