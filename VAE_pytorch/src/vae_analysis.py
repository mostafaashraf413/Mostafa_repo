#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:47:37 2021

@author: mostafa_shehata
"""

import cv2
import matplotlib.pyplot as plt
import utils
from vae_architecture import VAE
import torch
from os import listdir
from os.path import isfile, join
from random import sample



def random_evaluation():
    
    faces_dir = '../data/faces/img_align_celeba/'
    all_faces = [join(faces_dir, f) for f in listdir(faces_dir) if isfile(join(faces_dir, f))]
    input_imgs =  sample(all_faces, 5)
    
    for i in input_imgs:
        plt.imshow(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))
        plt.show()

    input_imgs = utils.process_in(input_imgs)
    print('input batch shape', input_imgs.shape)
    
    vae = VAE(k = 256)
    vae.load_state_dict(torch.load('../models/vae_faces.model'))
    # vae = vae.eval()
    
    decoded_imgs, mu_vec, log_var_vec = vae(input_imgs)
    # mu_vec, log_var_vec = vae.encode(input_imgs)
    # decoded_imgs = vae.decode(mu_vec)
    
    decoded_imgs = utils.process_out(decoded_imgs)
    
    for i in range(decoded_imgs.shape[0]):
        plt.imshow(cv2.cvtColor(decoded_imgs[i], cv2.COLOR_BGR2RGB))
        plt.show()   
        
        
if __name__ == '__main__':
    random_evaluation()