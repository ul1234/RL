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


landmarks = pd.read_csv('data/faces/face_landmarks.csv')

print(landmarks.iloc[[0,1]])

print('done')