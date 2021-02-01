#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:47:52 2021

@author: mostafa_shehata
"""

import cv2
import matplotlib.pyplot as plt
import torch

from os import listdir
from os.path import isfile, join
from random import sample 


def process_in(paths):
    batch = []
    for i in paths:
        img = cv2.imread(i)
        img = cv2.resize(img, (128, 128))         
        batch.append(img)
        
    batch = torch.tensor(batch, dtype=torch.float).permute(0,3,1,2)
    batch = batch/255 # scaling
    return batch



def process_out(imgs):
    
    imgs = imgs.permute(0,2,3,1)
    imgs = imgs * 255
    imgs = imgs.detach().numpy()
    return imgs    
    


class FacesGenerator:
    
    def __init__(self, faces_dir = '../data/faces/img_align_celeba', batch_size = 256, test_ratio = 0):
        self.faces_dir = faces_dir
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        
        self.all_faces = [join(self.faces_dir, f) for f in listdir(self.faces_dir) if isfile(join(self.faces_dir, f))]
        self.n_batches = (len(self.all_faces) // batch_size) + 1
        
    
    def generate_faces(self):
        for i in range(self.n_batches):
            batch = sample(self.all_faces, self.batch_size) 
            yield i, batch
        
        
        
        
        
if __name__ == '__main__':
    for i, batch in FacesGenerator().generate_faces():
        print(i)
        print('#######################')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        