import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # This defines the architecture of our CNN
        # 1. Input is a grayscale image
        # For the first convolutional layer lets take 32 channels/feature maps
        # Also lets take feature matrix as 5x5 square
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 2. Max pool layer
        # Kernel size = 2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)

        # Calculations for 2nd Convolutional layer
        # 32 inputs, 64 outputs, 5 feature matrix size
        # output size = (W - F)/ S + 1 = (115 - 5)/ 1 + 1 = 111
        # output tensor dimensions: (64, 55, 55)
        # after max pool dimensions: (64, 55, 55)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        # Calculations for 3rd Convolutional layer
        # 64 inputs, 128 outputs, 5 feature matrix size
        # output size = (W - F) / S + 1 = (55 - 5)/ 1 + 1 = 51
        # output tensor dimensions after max pool : (64, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv3_drop = nn.Dropout(p=0.2)
        # fully connected layer with 128 outputs * 25*25 filtered/pooled map size
        self.fc1 = nn.Linear(128*25*25, 2086)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(2086, 1024)
        self.fc2_drop = nn.Dropout(p=0.3)
        # fully connected 2nd layer with 2086 inputs and 2 outputs
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        # x is the input image
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        # before connecting to linear layer, we need to flatten
        x = x.view(x.size(0), -1)

        # Finally the linear layer
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        return x
