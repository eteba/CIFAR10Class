import os

import torch
from torch.utils.data import Dataset, DataLoader

# Define our custom dataset
# 1st byte: label
# next 32*32 bytes: red channel (row-wise ordered)
# next 32*32 bytes: green channel
# next 32*32 bytes: blue channel
# Image + label size: 3*32*32 + 1 = 3073 B
# GOAL is torch image: C x H x W
class CIFAR10Data(Dataset):
    def __init__(self, infile):
        # Open file and extract basic information:
        self.f_in = open(infile, "rb")
		
        self.nimgs = int(os.path.getsize(infile)/3073)
        
        # Read the whole file to RAM
        # Hacky way to reduce IO in systems without
        # a ram fs to store data.bin.
        self.buf = self.f_in.read(self.nimgs * 3073)
        
        
        print("(DATASET) Opened train dataset with ", self.nimgs, "images.")
    
    def __len__(self):
        return self.nimgs
    
    # From myData[0] to myData[nimgs-1]
    # Return a dictionary with the image and its label
    def __getitem__(self, idx):
        label = int.from_bytes(self.buf[idx*3073 + 0:idx*3073 + 1], byteorder="little")
        label_tensor = torch.zeros(10)
        label_tensor[label] = 1
        
        # Torch images: C x H x W
        image = torch.empty(3, 32, 32)
        
        for ch in range(3):
            for i in range(32):
                for j in range(32):
                    mybyte = self.buf[idx*3073 + 1 + 32*32*ch+32*i+j : idx*3073 + 1 + 32*32*ch+32*i+j + 1]
                    image[ch,i,j] = int.from_bytes(mybyte, byteorder="little") / 256
        
        item = {"image": image, "label": label_tensor}
        return item
