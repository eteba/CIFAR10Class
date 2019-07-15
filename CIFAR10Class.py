import os
import sys

import matplotlib.pyplot as plt
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from myNet import Net
from myDataset import CIFAR10Data
#from myDataset_readFullFile import CIFAR10Data

# Auxiliary plotting function
def plot_status(image, output, real_out, conv1_out, net, fig_counter):
    aux = [0,1,2,3,4,5,6,7,8,9]
    plt.ion()
    plt.close()
    
    fig = plt.figure(figsize=(13,8))
    gridsize = (4, 18)
    
    # net.conv1[0].weight.size(): (filt, in_ch, kernel_size, kernel_size)
    # conv1_out.size(): (batch_size, filt, 16, 16)
    
    # We have to re-scale weight matrixes to the [0,1] range.
    for filt in range(20):
        if filt < 10:
            plt.subplot2grid(gridsize, (0,filt))
        else:
            plt.subplot2grid(gridsize, (1,filt-10))
            
        plt.cla()
        weights_aux = net.conv1[0].weight[filt,:,:,:].permute(1,2,0).detach().numpy()
        weights_aux = weights_aux - weights_aux.min()
        weights_aux = weights_aux * 1/weights_aux.max()
        plt.imshow(weights_aux)
        plt.show()
        plt.pause(0.001)
        plt.axis("off")
    
    for filt in range(20):
        if filt < 10:
            plt.subplot2grid(gridsize, (2,filt))
        else:
            plt.subplot2grid(gridsize, (3,filt-10))
        
        plt.cla()
        plt.imshow(conv1_out.detach().numpy()[0,filt,:,:])
        plt.show()
        plt.pause(0.001)
        plt.axis("off")
    
    plt.subplot2grid(gridsize, (0,10), rowspan=1, colspan=8)
    plt.cla()
    image_aux = image[0,:,:,:].permute(1,2,0).detach().numpy()
    plt.imshow(image_aux)
    plt.show()
    plt.pause(0.001)
    plt.axis("off")
    
    plt.subplot2grid(gridsize, (1,10), rowspan=3, colspan=8)
    plt.cla()
    output_aux = output[0,:].detach().numpy()
    plt.hist(aux, 10, weights=output_aux, alpha=0.5)
    plt.hist(aux, 10, weights=real_out, alpha=0.5)
    plt.show()
    plt.pause(0.001)
    plt.axis("off")
        
    savepath = "fig_{}.png".format(fig_counter)
    plt.savefig(savepath)


#
# Main program
#

# Open training data
myData = CIFAR10Data("data/data.bin")

# Split the data in training and evaluating sets
train_dataset, eval_dataset = random_split(myData, [45000, 5000])

# Create a DataLoader
# NOTE: num_workers have to be 1 to avoid errors in CIFAR10Data.__getitem__()
#       due to parallel accesses to the same file.
dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
eval_dataloader = DataLoader(eval_dataset, batch_size=20, shuffle=False, num_workers=0)

# Create the network
mynet = Net()
print(mynet)

# Loss function
criterion = nn.MSELoss()

# Optimizer to avoid one-by-one weight updating
optimizer = torch.optim.SGD(mynet.parameters(), lr=0.1, momentum=0.9)

# Loop for some epochs
fig_counter = 1
for epoch in range(20):
    # Simple learning rate decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1/(1 + epoch)
    
    cummulated_loss = 0.0
    for i, data_batched in enumerate(dataloader):
        # Set the model to train mode
        mynet.train()
        
        # Get the image and its label
        images_t = data_batched["image"]
        labels_t = data_batched["label"]
        
        # Input and target information is read.
        # Process input through mynet.
        optimizer.zero_grad()
        outputs = mynet(images_t)
        
        # Compute loss, back-propagate and update weights
        loss = criterion(outputs, labels_t)
        loss.backward()
        optimizer.step()
        
        cummulated_loss += loss.item()
        
        # Visual evaluation of the network performance using
        # the evaluation dataset.
        if i % 250 == 249:
            print("Loss is ", loss.item(), " after ", (i+1)*20, " images processed.")
            
            # Re-apply the first layer to a random input to
            # get the convolved images.
            test_image = eval_dataset[randint(0, len(eval_dataset)-1)]
            real_out = test_image["label"]
            test_image = test_image["image"]
            test_image = test_image.unsqueeze(0)    # Add a null batch dimension
            
            # Set the model to eval mode and compute its output
            mynet.eval()
            with torch.no_grad():
                conv1_out = mynet.conv1(test_image)
                net_out = mynet(test_image)
            
            plot_status(test_image, net_out, real_out, conv1_out, mynet, fig_counter)
            fig_counter += 1
        
    # Evaluate the cummulated loss in the evaluation set.
    print("(EVALUATION) Computing cummulated loss in the evaluation subset...")
        
    cummulated_loss_e = 0.0
    for i, data_batched in enumerate(eval_dataloader):
        mynet.eval()
        images_e = data_batched["image"]
        labels_e = data_batched["label"]
        
        outputs_e = mynet(images_e)
        
        loss_e = criterion(outputs_e, labels_e)
        cummulated_loss_e += loss_e.item()
        
    print("(EPOCH) Epoch", epoch, "completed.")
    print("(EPOCH) Training cummulated loss:  ", cummulated_loss)
    print("(EPOCH) Evaluation cummulated loss:", cummulated_loss_e)
    print("\n")
    
    torch.save(mynet.state_dict(), "out/model/saved_model_e{}".format(epoch))

print("Finished training.")
