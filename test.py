import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import numpy
import sys

class test(nn.Module):
    def __init__(self, planes):
        super(test, self).__init__()

        self.planes = planes
        self.weight1 = torch.ones(1,1,3,3)
        self.weight2 = torch.randn(1,1,3,3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = None
        for i in range (self.planes):
            if i == 0:
                out = F.conv2d(x[:,i,:,:].unsqueeze(1), self.weight1, stride=1, padding=1)
            else:
                temp_out = F.conv2d(x[:,i,:,:].unsqueeze(1), self.weight1, stride=1, padding=1)
                out = torch.cat(([out, temp_out]), dim=1)
        out = self.bn1(out)
        for i in range (self.planes):
            if i == 0:
                out = F.conv2d(x[:,i,:,:].unsqueeze(1), self.weight2, stride=1, padding=1)
            else:
                temp_out = F.conv2d(x[:,i,:,:].unsqueeze(1), self.weight2, stride=1, padding=1)
                out = torch.cat(([out, temp_out]), dim=1)
        out = self.bn2(out)

        return out





if __name__ == "__main__":

    input1 = torch.ones(1,5,10,10)
    #  print(input1)
    planes = input1.shape[1]
    #  print(input1[:,0,:,:])
    model = test(planes)
    output = model(input1)
    print("Total Output ")
    print(output)
    print(output.size())
    for i in range(planes):
        print("=========== for channel {}===============".format(i))
        print(output[:,i,:,:].unsqueeze(1))
        print(output[:,i,:,:].unsqueeze(1).size())

