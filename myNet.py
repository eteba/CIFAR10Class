import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Network layout:
        # Input: 3x32x32.
        # 1st layer: 20 x conv2d(3x3) -> 20x32x32
        # 2nd layer: 30 x conv2d(3x3) + MaxPool2d(2) -> 30x16x16
        # 2nd layer: 40 x conv2d(3x3) + MaxPool2d(3) -> 40x6x6
        # 3rd layer: Fully connected(40*6*6 -> 100)
        # 4th layer: Fully connected(100 -> 10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.15))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=0.25))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(30, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout2d(p=0.35))
        
        self.fc1 = nn.Sequential(
            nn.Linear(40*6*6, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        
        self.linout = nn.Linear(100, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(-1, 1440) # 1440 = 40*6*6
        
        x = self.fc1(x)
        x = self.linout(x)
        return F.softmax(x, dim=-1)
