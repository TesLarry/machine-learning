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
import shutil
import random

from lxml import etree


# split data
class pascal_voc_data(object):

    # initialization
    def __init__(self, voc_root):
        self.img_root = os.path.join(voc_root, "JPEGImages")
        self.ann_root = os.path.join(voc_root, "Annotations")
        self.cls_root = os.path.join(voc_root, "ImageSets", "Main")
        self.new_root = os.path.join(voc_root, "..//", "VOC_image_cls")
        self.cls_file = ['train', 'valid']
        self.cat_dict = self.category_index("pascal_voc_classes.json")
        self.xml_info = self.xml_reader()
        self.make_dir()

    # read class_indict
    def category_index(self, json_name):
        category_index = {}
        try:
            json_path = os.path.join(os.getcwd(), json_name)
            json_file = open(json_path, 'r')
            class_dict = json.load(json_file)
            category_index = {v: k for k, v in class_dict.items()}
            return category_index
        except Exception as exception:
            print(exception)
            exit(-1)
    
    # reading xml files
    def xml_reader(self):
        txt_files = os.path.join(self.cls_root, "trainval.txt")
        with open(txt_files) as read:
            xml_list = [os.path.join(self.ann_root, line.strip() + ".xml") 
                        for line in read.readlines()]
        xml_info = {}
        for i, xml_path in enumerate(xml_list):
            with open(xml_path) as fid:
                xml_str = fid.read()
            data = self.parse_xml_to_dict(etree.fromstring(xml_str))
            key  = data['annotation']['filename']
            if len(data['annotation']['object']) == 1:
                val = data['annotation']['object'][0]['name']
            else:  # find the object with the largest size
                val = ""
                square_size = 0
                for item in data['annotation']['object']:
                    bbox = item['bndbox']
                    w = int(bbox['xmax']) - int(bbox['xmin'])
                    h = int(bbox['ymax']) - int(bbox['ymin'])
                    if w * h >= square_size:
                        val = item['name']
                        square_size = w * h
            xml_info[key] = val
            print("\rcomputing [{}/{}]".format(i + 1, len(xml_list)), end = "")
        return xml_info
    
    # convert xml to dict
    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        else:
            result = {}
            for child in xml:
                child_result = self.parse_xml_to_dict(child)
                if child.tag != 'object':
                    result[child.tag] = child_result[child.tag]
                else:
                    if child.tag not in result:
                        result[child.tag] = []
                    result[child.tag].append(child_result[child.tag])
            return {xml.tag: result}
    
    # split data
    def split(self, train_rate):
        # sampling
        itr = 0
        img = list(self.xml_info.keys())
        train_img = random.sample(img, k = int(len(img) * train_rate))
        print("\n")
        for key, value in self.xml_info.items():
            itr = itr + 1
            if key in train_img:
                old_path = os.path.join(self.img_root, key)
                new_path = os.path.join(self.new_root, self.cls_file[0], value)
                shutil.copy(old_path, new_path)
            else:
                old_path = os.path.join(self.img_root, key)
                new_path = os.path.join(self.new_root, self.cls_file[1], value)
                shutil.copy(old_path, new_path)
            print("\rcopying [{}/{}]".format(itr, len(img)), end = "")
        
    
    # make direction path and move files
    def make_dir(self):
        for key, value in self.cat_dict.items():
            # make dir
            for name in self.cls_file:
                path = os.path.join(self.new_root, name, value)
                if not os.path.exists(path):
                    os.makedirs(path)


if __name__ == "__main__":
    
    train_rate = 0.8
    voc_root = 'd:\Data Set\image_dataset\VOCdevkit\VOC2012'
    voc_data = pascal_voc_data(voc_root)
    voc_data.split(train_rate)
