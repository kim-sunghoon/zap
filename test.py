import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import Config as cfg
import numpy
import sys

class test(nn.Module):
    def __init__(self, planes):
        super(test, self).__init__()

        self.planes = planes
        if cfg.filter_mode == 0:
            self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=planes)
        elif cfg.filter_mode in [1,2,4,8]:
            self.conv1 = nn.Conv2d(cfg.filter_mode, cfg.filter_mode, kernel_size=3, padding=1, stride=1, groups=cfg.filter_mode)
            self.conv2 = nn.Conv2d(cfg.filter_mode, cfg.filter_mode, kernel_size=3, padding=1, stride=1, groups=cfg.filter_mode)
        else:
            raise NotImplementedError


        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        out = None
        if cfg.filter_mode == 0:
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            return out

        elif cfg.filter_mode in [1,2,4,8]:
            out1 = None
            out2 = None
            for i in range(0, self.planes, cfg.filter_mode):
                if i == 0:
                    out1 = self.conv1(x[:, i:i+cfg.filter_mode,:,:])
                else:
                    temp_out1 = self.conv1(x[:,i:i+cfg.filter_mode,:,:])
                    out1 = torch.cat(([out1, temp_out1]), dim=1)
            out1 = self.bn1(out1)
            out1 = F.relu(out1)

            for i in range(0, self.planes, cfg.filter_mode):
                if i == 0:
                    out2 = self.conv1(out1[:, i:i+cfg.filter_mode,:,:])
                else:
                    temp_out2 = self.conv1(out1[:,i:i+cfg.filter_mode,:,:])
                    out2 = torch.cat(([out2, temp_out2]), dim=1)
            out2 = self.bn2(out2)

            return out2
        else:
            raise NotImplementedError






if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (8,10,10)

    input1 = torch.ones(input_size).unsqueeze(0).to(device)
    print(input1)
    print(input1.size())
    print("==="*20)
    #  print(input1)
    planes = input1.shape[1]
    #  print(input1[:,0,:,:])
    model = test(planes)
    model = model.to(device)
    print("="*20 + "  Model View " + "="*20)
    print(model)
    #  torchsummary.summary(model, (8,10,10))
    output = model(input1)
    print("Total Output ")
    print(output)
    print(output.size())
    for i in range(planes):
        print("=========== for channel {}===============".format(i))
        print(output[:,i,:,:].unsqueeze(1))
        print(output[:,i,:,:].unsqueeze(1).size())

