#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


plt.ion()


def show_img(img, landmarks):
    #plt.imshow(img)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, c = 'r', marker = '*')

#plt.figure()
#show_img(landmarks_frame, 0)
#plt.ioff()
#plt.show()
class FacesDataset(Dataset):
    def __init__(self, landmark_file, transform = None):
        self.landmark_frame = pd.read_csv(landmark_file)
        self.file_path = os.path.dirname(landmark_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, idx):
        img_name = self.landmark_frame.iloc[idx, 0]
        landmarks = self.landmark_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype(np.float32).reshape(-1, 2)
        img_file = os.path.join(self.file_path, img_name)
        img = io.imread(img_file)
        sample = {'img': img, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Scale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, landmarks = sample['img'], sample['landmarks']
        h, w, layers = img.shape
        factor = self.output_size / min([h, w])
        new_h, new_w = int(h * factor), int(w * factor)
        new_img = transform.resize(img, (new_h, new_w))
        landmarks = landmarks * factor
        return {'img': new_img, 'landmarks': landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        img, landmarks = sample['img'], sample['landmarks']
        x = np.random.randint(0, img.shape[0] - self.output_size[0])
        y = np.random.randint(0, img.shape[1] - self.output_size[1])
        new_img = img[x:x+self.output_size[0], y:y+self.output_size[1]]
        landmarks[:, 0] -= y
        landmarks[:, 1] -= x
        return {'img': new_img, 'landmarks': landmarks}

class ToTensor(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        img, landmarks = sample['img'], sample['landmarks']
        img = img.transpose([2, 0, 1])
        return {'img': torch.from_numpy(img), 'landmarks': torch.from_numpy(landmarks)}

compse = transforms.Compose([Scale(300), RandomCrop((200, 200)), ToTensor()])
dataset = FacesDataset('data/faces/face_landmarks.csv', transform = compse)
plt.figure()
for i in range(len(dataset)):
    sample = dataset[i]
    print(i, sample['img'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i+1)
    ax.axis('off')
    ax.set_title('sample {}'.format(i))
    plt.tight_layout()
    show_img(**sample)
    if i == 3: break
plt.ioff()
plt.show()
print('done')