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
    def __init__(self, k = 256):
        
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 64)
        self.batch_norm_0 = nn.BatchNorm2d(num_features = 32)
        
        self.conv_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 32)
        self.batch_norm_1 = nn.BatchNorm2d(num_features = 64)
        
        self.conv_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 16)
        self.batch_norm_2 = nn.BatchNorm2d(num_features = 64)
        
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 8)
        self.batch_norm_3 = nn.BatchNorm2d(num_features = 64)
        
        self.mu = nn.Linear(9216, k)
        self.log_var = nn.Linear(9216, k)
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.2)
            
        
    def forward(self, x):
        
        x = self.conv_0(x)
        x = self.batch_norm_0(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = x.view(x.shape[0], -1) # flatten shape is 9216
        
        mu_vec = self.mu(x)
        log_var_vec = self.log_var(x)
        
        return mu_vec, log_var_vec 
    
    
#class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv_0 = torch.nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 64
#
#    def forward(self, x):

        
        
        
        
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    
    imgs = []
    for i in ['000011.jpg', '000001.jpg', '000002.jpg', '000003.jpg']:
        img = cv2.imread('../data/faces/img_align_celeba/'+i )
        img = cv2.resize(img, (128, 128))  
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        imgs.append(img)


    input_imgs = torch.FloatTensor(imgs).permute(0,3,1,2)
    print('input batch shape', input_imgs.shape)
    
    enc = Encoder()
    mu_vec, log_var_vec = enc(input_imgs)
    print('output mu shape', mu_vec.shape)
    print('output lgo_var shape', log_var_vec.shape)
    
#    
#    encoded_img = encoded_img.permute(0,2,3,1).detach().numpy()[0]
#    plt.imshow(encoded_img)
#    plt.show()
    
    
    
    
    
    
    
    