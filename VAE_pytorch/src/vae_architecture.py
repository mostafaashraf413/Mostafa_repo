#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:20:33 2021

@author: mostafa_shehata
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal

norm_dist = Normal(0, 1)
def sample_from(mu, log_var):
    epsilon = norm_dist.sample((mu.shape[0], mu.shape[1])).to(mu.device)
    sample = mu + (torch.exp(log_var / 2) * epsilon)
    return sample
    
    

class VAE(nn.Module):
    def __init__(self, k):
        
        super().__init__()
        
        # encoder
        self.enc_conv_0 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 64)
        self.enc_batch_norm_0 = nn.BatchNorm2d(num_features = 32)
        
        self.enc_conv_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 32)
        self.enc_batch_norm_1 = nn.BatchNorm2d(num_features = 64)
        
        self.enc_conv_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 16)
        self.enc_batch_norm_2 = nn.BatchNorm2d(num_features = 64)
        
        self.enc_conv_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 8)
        self.enc_batch_norm_3 = nn.BatchNorm2d(num_features = 64)
        
        self.mu = nn.Linear(9216, k)
        self.log_var = nn.Linear(9216, k)
        
        #decoder
        self.dec_l_0 = nn.Linear(k, 9216)
        
        self.dec_conv_3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 8)
        self.dec_batch_norm_3 = nn.BatchNorm2d(num_features = 64)
        
        self.dec_conv_2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 16)
        self.dec_batch_norm_2 = nn.BatchNorm2d(num_features = 64)
        
        self.dec_conv_1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 32)
        self.dec_batch_norm_1 = nn.BatchNorm2d(num_features = 32)
        
        self.dec_conv_0 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 64)
        self.dec_batch_norm_0 = nn.BatchNorm2d(num_features = 3)
        
        # functions
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.25)
        self.sigmoid = nn.Sigmoid()
        
            
        
    def encode(self, x):
        x = self.enc_conv_0(x)
        x = self.enc_batch_norm_0(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.enc_conv_1(x)
        x = self.enc_batch_norm_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.enc_conv_2(x)
        x = self.enc_batch_norm_2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.enc_conv_3(x)
        x = self.enc_batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = x.reshape(x.shape[0], -1) # flatten shape is 9216
        
        mu_vec = self.mu(x)
        log_var_vec = self.log_var(x)
        
        return mu_vec, log_var_vec 
    
    def decode(self, x):
        
        x = self.dec_l_0(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], 64, 12, 12)
        
        x = self.dec_conv_3(x)
        x = self.dec_batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dec_conv_2(x)
        x = self.dec_batch_norm_2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dec_conv_1(x)
        x = self.dec_batch_norm_1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dec_conv_0(x)
        x = self.dec_batch_norm_0(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.sigmoid(x)
        
        return x
    
    def forward(self, x):
        mu_vec, log_var_vec = self.encode(x)
        sampled = sample_from(mu_vec, log_var_vec)
        reconstructed = self.decode(sampled)
        return reconstructed, mu_vec, log_var_vec
  
        
        
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import utils
    
    input_imgs =  ['../data/faces/img_align_celeba/'+i for i in ['000011.jpg', '000001.jpg', 
                                                                 '000002.jpg', '000003.jpg']]
    
    for i in input_imgs:
        plt.imshow(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))
        plt.show()

    input_imgs = utils.process_in(input_imgs)
    print('input batch shape', input_imgs.shape)
    
    # vae = VAE(k = 256)
    decoded_imgs, mu_vec, log_var_vec = vae(input_imgs.to('cuda'))
    print('output mu shape', mu_vec.shape)
    print('output lgo_var shape', log_var_vec.shape)
    
    decoded_imgs = utils.process_out(decoded_imgs.cpu())
    
    for i in range(decoded_imgs.shape[0]):
        # plt.imshow(cv2.cvtColor(decoded_imgs[i], cv2.COLOR_BGR2RGB))
        plt.imshow(decoded_imgs[i])
        plt.show()        
    
    
    
    
    
    
    
    
    
    