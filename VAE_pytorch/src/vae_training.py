#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:47:21 2021

@author: mostafa_shehata
"""
import torch
import torch.nn as nn
import torch.optim as optim
from vae_architecture import VAE
from utils import FacesGenerator, process_in
from datetime import datetime

# parameters:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

loss_criterion = nn.MSELoss(reduction = 'mean')
r_loss_factor = 1e4
k = 256
vae = VAE(k = k).to(device)
lr = 5e-4
optimizer = optim.Adam(vae.parameters(), lr = lr)
n_epochs = 200
batch_size = 32


def vae_kl_loss(mu, log_var):
    kl_loss = -0.5 * torch.sum((1 + log_var - torch.square(mu) - torch.exp(log_var))) #, dim = 1)
    return kl_loss


def vae_loss(y_true, y_pred, mu, log_var):
    const_loss = loss_criterion(y_pred, y_true)
    kl_loss = vae_kl_loss(mu, log_var) #.unsqueeze(1)
    total_loss = (r_loss_factor * const_loss) + kl_loss
    return total_loss


def train():
    
    
    face_gen = FacesGenerator(batch_size = batch_size) 
    
    print(datetime.now().time(), 'VAE training has been started')
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_num, batch in face_gen.generate_faces():
            batch = process_in(batch).to(device)
            
            rec_imgs, mu_vec, log_var_vec = vae(batch)
            total_loss = vae_loss(y_true = batch, y_pred = rec_imgs, mu = mu_vec, log_var = log_var_vec)
            epoch_loss += total_loss.item()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if batch_num % 100 == 0:
                print(datetime.now().time(), ' # epoch %d, batch %d, batch_loss = %f'%(epoch, batch_num, total_loss.item()))
            
        avg_epoch_loss = epoch_loss/(batch_num+1)
        print(datetime.now().time(), ' +--> epoch %d: training_loss = %f'%(epoch, avg_epoch_loss))
        torch.save(vae.state_dict(), '../models/vae_faces.model')



if __name__ == '__main__':
    
    train()
    
    # testing vae loss
    # kl_loss = vae_kl_loss(torch.tensor([[0,0,0], [-1,1,1]], dtype=torch.float),
    #                       torch.tensor([[1,1,1], [1.1,1,1]], dtype=torch.float))
    # print('kl_loss  = ', kl_loss)
    
    # vae_total_loss = vae_loss(y_true = torch.tensor([[2],[2]], dtype = torch.float),
    #                           y_pred = torch.tensor([[2],[2]], dtype = torch.float),
    #                           mu = torch.tensor([[0,0,0], [-1,1,1]], dtype=torch.float),
    #                           log_var = torch.tensor([[1,1,1], [1.1,1,1]], dtype=torch.float))
    # print('total_loss = ', vae_total_loss)