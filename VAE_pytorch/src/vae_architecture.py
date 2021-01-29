#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:20:33 2021

@author: mostafa_shehata
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 64)
        self.batch_norm_0 = nn.BatchNorm2d(num_features = 32)
        
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d()
            
        
    def forward(self, x):
        x = self.conv_0(x)
        x = self.batch_norm_0(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 64

    def forward(self, x):

        
        
        
        
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    
    
    img = cv2.imread('../data/faces/img_align_celeba/000011.jpg' )
    img = cv2.resize(img, (128, 128))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    enc = Encoder()
    encoded_img = torch.FloatTensor([img]).permute(0,3,1,2)
    encoded_img = enc(encoded_img)
    
#    
#    encoded_img = encoded_img.permute(0,2,3,1).detach().numpy()[0]
#    plt.imshow(encoded_img)
#    plt.show()