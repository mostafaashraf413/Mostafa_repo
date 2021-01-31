#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:47:52 2021

@author: mostafa_shehata
"""

import cv2
import matplotlib.pyplot as plt
import torch


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
    