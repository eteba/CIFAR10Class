import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from myNet import Net
from myDataset import CIFAR10Data

#
# Main program
#

# From an initial random image, we are creating a loss
# function with the goal of maximizing each one of the filters
# in each convolutional layer. The gradient ascent process
# gives us an input image which activates the most each filter.
in_size = 128
tol = 1e-5  # 0.001%

input_image = torch.empty(20,3,in_size,in_size, requires_grad=False)
input_image = input_image.unsqueeze(0)

input_image2 = torch.empty(30,3,in_size,in_size, requires_grad=False)
input_image2 = input_image2.unsqueeze(0)

input_image3 = torch.empty(40,3,in_size,in_size, requires_grad=False)
input_image3 = input_image3.unsqueeze(0)


# Start maximization of conv1 filters.
for f_idx in range(20):
    print("conv1 - Filter {}".format(f_idx))

    # Initialize the first input image to grey + random noise
    with torch.no_grad():
        for i in range(in_size):
            for j in range(in_size):
                for ch in range(3):
                    value = random.normal(0.5, 0.2)
                    if value < 0:
                        value = 0
                    if value > 1:
                        value = 1
                    
                    input_image[0,f_idx,ch,i,j] = value
    
    input_image.requires_grad_(True)
    
    # Load our trained model
    mynet = Net()
    mynet.load_state_dict(torch.load("out/models/saved_model_e19"))
    mynet.eval()
    
    # Maximize the mean value of all the elements of the filter
    # output.
    old_norm_loss = -1
    it = 1
    while True:
        out_conv1 = mynet.conv1(input_image[:,f_idx,:,:,:])
        loss = torch.mean(out_conv1[0,f_idx,:,:])
        
        loss.backward()
        
        # Gradient ascent.
        with torch.no_grad():
            grad = input_image.grad[:,f_idx,:,:,:]
            grad /= (torch.sqrt(torch.mean(grad**2)) + 1e-5)
            input_image[:,f_idx,:,:,:] += grad*5e-2
            input_image.grad.zero_()
            
            # Normalization of the image just using multiplicative factors
            # boosts the process
            input_image[:,f_idx,:,:,:] *= 1/max(abs(input_image[:,f_idx,:,:,:].max()), abs(input_image[:,f_idx,:,:,:].min()))
            
            # In order to converge, we evaluate the loss in a normalized
            # copy of the input image.
            aux_input_image = input_image[:,f_idx,:,:,:].data.clone()
            aux_min = aux_input_image.min()
            aux_input_image -= aux_input_image.min()
            aux_input_image *= 1/aux_input_image.max()
            aux_out = mynet.conv1(aux_input_image)
            norm_loss = torch.mean(aux_out[0,f_idx,:,:]) + tol*1e-3
            diff_loss = abs(norm_loss - old_norm_loss)/old_norm_loss
            
            old_norm_loss = norm_loss
            
        if abs(diff_loss) < tol:
            break
        
        if it > 1e3:
            print("(DEBUG) conv1 - Filter {}: There is no convergence or it is hard to find it.".format(f_idx))
            break
        
        it += 1
    
    # Normalize the image to fit in the [0-1] range so we can plot it
    with torch.no_grad():
        input_image[:,f_idx,:,:,:] -= input_image[:,f_idx,:,:,:].min()
        input_image[:,f_idx,:,:,:] *= 1/input_image[:,f_idx,:,:,:].max()
    
# Start maximization of conv2 filters.
for f_idx in range(30):
    print("conv2 - Filter {}".format(f_idx))

    # Initialize the first input image to grey + random noise
    with torch.no_grad():
        for i in range(in_size):
            for j in range(in_size):
                for ch in range(3):
                    value = random.normal(0.5, 0.2)
                    if value < 0:
                        value = 0
                    if value > 1:
                        value = 1
                    
                    input_image2[0,f_idx,ch,i,j] = value
    
    input_image2.requires_grad_(True)

    # Load our trained model
    mynet = Net()
    mynet.load_state_dict(torch.load("out/models/saved_model_e19"))
    mynet.eval()

    # Maximize the mean value of all the elements of the filter
    # output.
    old_norm_loss = -1
    it = 1
    while True:
        out_conv2 = mynet.conv1(input_image2[:,f_idx,:,:,:])
        out_conv2 = mynet.conv2(out_conv2)
        loss = torch.mean(out_conv2[0,f_idx,:,:])
        
        loss.backward()
        
        with torch.no_grad():
            grad = input_image2.grad[:,f_idx,:,:,:]
            grad /= (torch.sqrt(torch.mean(grad**2)) + 1e-5)
            input_image2[:,f_idx,:,:,:] += grad*5e-2
            input_image2.grad.zero_()
            
            # Normalization of the image just using multiplicative factors
            # boosts the process
            input_image2[:,f_idx,:,:,:] *= 1/max(abs(input_image2[:,f_idx,:,:,:].max()), abs(input_image2[:,f_idx,:,:,:].min()))
            
            # In order to converge, we evaluate the loss in a normalized
            # copy of the input image.
            aux_input_image = input_image2[:,f_idx,:,:,:].data.clone()
            aux_min = aux_input_image.min()
            aux_input_image -= aux_input_image.min()
            aux_input_image *= 1/aux_input_image.max()
            aux_out = mynet.conv1(aux_input_image)
            aux_out = mynet.conv2(aux_out)
            norm_loss = torch.mean(aux_out[0,f_idx,:,:]) + tol*1e-3
            diff_loss = abs(norm_loss - old_norm_loss)/old_norm_loss
            
            old_norm_loss = norm_loss
            
        if abs(diff_loss) < tol:
            break
        
        if it > 1e3:
            print("(DEBUG) conv2 - Filter {}: There is no convergence or it is hard to find it.".format(f_idx))
            break

        it += 1

        
    with torch.no_grad():
        input_image2[:,f_idx,:,:,:] -= input_image2[:,f_idx,:,:,:].min()
        input_image2[:,f_idx,:,:,:] *= 1/input_image2[:,f_idx,:,:,:].max()


# Start maximization of conv3 filters.
for f_idx in range(40):
    print("conv3 - Filter {}".format(f_idx))

    # Initialize the first input image to grey + random noise
    with torch.no_grad():
        for i in range(in_size):
            for j in range(in_size):
                for ch in range(3):
                    value = random.normal(0.5, 0.2)
                    if value < 0:
                        value = 0
                    if value > 1:
                        value = 1
                    
                    input_image3[0,f_idx,ch,i,j] = value
    
    input_image3.requires_grad_(True)

    # Load our trained model
    mynet = Net()
    mynet.load_state_dict(torch.load("out/models/saved_model_e19"))
    mynet.eval()

    # Maximize the mean value of all the elements of the filter
    # output.
    old_norm_loss = -1
    it = 1
    while True:
        out_conv3 = mynet.conv1(input_image3[:,f_idx,:,:,:])
        out_conv3 = mynet.conv2(out_conv3)
        out_conv3 = mynet.conv3(out_conv3)
        loss = torch.mean(out_conv3[0,f_idx,:,:])
        
        loss.backward()
        
        with torch.no_grad():
            grad = input_image3.grad[:,f_idx,:,:,:]
            grad /= (torch.sqrt(torch.mean(grad**2)) + 1e-5)
            input_image3[:,f_idx,:,:,:] += grad*5e-2
            input_image3.grad.zero_()
            
            # Normalization of the image just using multiplicative factors
            # boosts the process
            input_image3[:,f_idx,:,:,:] *= 1/max(abs(input_image3[:,f_idx,:,:,:].max()), abs(input_image3[:,f_idx,:,:,:].min()))
            
            # In order to converge, we evaluate the loss in a normalized
            # copy of the input image.
            aux_input_image = input_image3[:,f_idx,:,:,:].data.clone()
            aux_min = aux_input_image.min()
            aux_input_image -= aux_input_image.min()
            aux_input_image *= 1/aux_input_image.max()
            aux_out = mynet.conv1(aux_input_image)
            aux_out = mynet.conv2(aux_out)
            aux_out = mynet.conv3(aux_out)
            norm_loss = torch.mean(aux_out[0,f_idx,:,:]) + tol*1e-3
            diff_loss = abs(norm_loss - old_norm_loss)/old_norm_loss
            
            old_norm_loss = norm_loss
            
        if abs(diff_loss) < tol:
            break

        if it > 1e3:
            print("(DEBUG) conv3 - Filter {}: There is no convergence or it is hard to find it.".format(f_idx))
            break
        
        it += 1

        
    # Normalize the image to fit in the [0-1] range so we can plot it
    with torch.no_grad():
        input_image3[:,f_idx,:,:,:] -= input_image3[:,f_idx,:,:,:].min()
        input_image3[:,f_idx,:,:,:] *= 1/input_image3[:,f_idx,:,:,:].max()


# Plotting
plt.ion()
fig = plt.figure(figsize=(20,8))
gridsize = (5, 20)

for f_idx in range(20):
    plt.subplot2grid(gridsize, (0,f_idx))
    plt.cla()
    
    plt.imshow(input_image[0,f_idx,:,:,:].permute(1,2,0).detach().numpy())
    plt.show()
    plt.pause(0.001)
    plt.axis("off")


for f_idx in range(20):
    plt.subplot2grid(gridsize, (1,f_idx))
    plt.cla()
    
    plt.imshow(input_image2[0,f_idx,:,:,:].permute(1,2,0).detach().numpy())
    plt.show()
    plt.pause(0.001)
    plt.axis("off")

for f_idx in range(10):
    plt.subplot2grid(gridsize, (2,f_idx))
    plt.cla()
    
    plt.imshow(input_image2[0,20+f_idx,:,:,:].permute(1,2,0).detach().numpy())
    plt.show()
    plt.pause(0.001)
    plt.axis("off")


for f_idx in range(20):
    plt.subplot2grid(gridsize, (3,f_idx))
    plt.cla()
    
    plt.imshow(input_image3[0,f_idx,:,:,:].permute(1,2,0).detach().numpy())
    plt.show()
    plt.pause(0.001)
    plt.axis("off")

for f_idx in range(20):
    plt.subplot2grid(gridsize, (4,f_idx))
    plt.cla()
    
    plt.imshow(input_image3[0,20+f_idx,:,:,:].permute(1,2,0).detach().numpy())
    plt.show()
    plt.pause(0.001)
    plt.axis("off")

# Save the image
plt.savefig("out/utils/feature_maps.png")
