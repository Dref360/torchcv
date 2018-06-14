from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import json
import numpy as np

class ListDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None, is_valid=False):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.classes = ["articulated_truck", "bicycle", "bus", "car", "motorcycle", 'motorized_vehicle', "non-motorized_vehicle",
                        "pedestrian", "pickup_truck", "single_unit_truck", "work_van"]

        data = json.load(open(list_file, 'r'))
        # v is [mio_id,items] we do not use mio_id yet.
        # data = [(pjoin(self.filepath, 'images', k+'.jpg'), v[1]) for k, v in data.items()]
        items = list(data.items())
        np.random.seed(300)
        np.random.shuffle(items)
        if is_valid:
            items = items[int(0.9 * len(items)):]
        else:
            items = items[:int(0.9 * len(items))]
        self.num_samples = len(items)
        for k, [_, vals] in items:
            self.fnames.append(k + '.jpg')
            box = []
            label = []
            ori = []
            parked = []
            for b in vals:
                # float(xmin), float(ymin), float(xmax), float(ymax)
                box.append([float(b['xmin']), float(b['ymin']), float(b['xmax']), float(b['ymax'])])
                label.append(self.classes.index(normalize_classes(b['class'])))
                ori.append(float(b['angle']) / (np.pi * 2))
                parked.append(int(float(b['mag']) < 1))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels

    def __len__(self):
        return self.num_samples

def normalize_classes(cls):
    cls = cls.lower()
    dat = {'pickup truck': 'pickup_truck',
           'pickuptruck': 'pickup_truck',
           'articulated truck': 'articulated_truck',
           'articulatedtruck': 'articulated_truck',
           'non-motorized vehicle': 'non-motorized_vehicle',
           'non-motorizedvehicle': 'non-motorized_vehicle',
           'nonmotorizedvehicle': 'non-motorized_vehicle',
           'motorized vehicle': 'motorized_vehicle',
           'motorizedvehicle': 'motorized_vehicle',
           'single unit truck': 'single_unit_truck',
           'singleunittruck': 'single_unit_truck',
           'work van': 'work_van', 'suv': 'car', 'minivan': 'car', 'workvan': 'work_van'}
    return dat[cls] if cls in dat else cls
