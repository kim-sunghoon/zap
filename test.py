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
        elif cfg.filter_mode == 1:
            self.weight1 = torch.ones((1,cfg.filter_mode,3,3), requires_grad = True).to("cuda")
            self.weight2 = torch.ones((1,cfg.filter_mode,3,3), requires_grad = True).to("cuda")
            print(self.weight1.size)
            #  self.weight1 = torch.ones((cfg.filter_mode,1,3,3), requires_grad = True).to("cuda")
            #  self.weight2 = torch.ones((cfg.filter_mode,1,3,3), requires_grad = True).to("cuda")
        elif cfg.filter_mode not in [2,4,8]:
            raise NotImplementedError
        else:
            self.conv1 = nn.Conv2d(cfg.filter_mode, cfg.filter_mode, kernel_size=3, padding=1, stride=1, groups=cfg.filter_mode)
            self.conv2 = nn.Conv2d(cfg.filter_mode, cfg.filter_mode, kernel_size=3, padding=1, stride=1, groups=cfg.filter_mode)



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


        elif cfg.filter_mode == 1:
            for i in range(0, self.planes, cfg.filter_mode):
                if i == 0:
                    out = F.conv2d(x[:, i:i+cfg.filter_mode,:,:], self.weight1, stride=1, padding=1, groups=cfg.filter_mode)
                else:
                    temp_out = F.conv2d(x[:,i:i+cfg.filter_mode,:,:], self.weight1, stride=1, padding=1, groups=cfg.filter_mode)
                    out = torch.cat(([out, temp_out]), dim=1)

            #  out = self.bn1(out)
            #
            #  for i in range(0, self.planes, cfg.filter_mode):
            #      if i == 0:
            #          out = F.conv2d(x[:,i:i+cfg.filter_mode,:,:].unsqueeze(1), self.weight2, stride=1, padding=1)
            #      else:
            #          temp_out = F.conv2d(x[:,i:i+cfg.filter_mode,:,:].unsqueeze(1), self.weight2, stride=1, padding=1)
            #          out = torch.cat(([out, temp_out]), dim=1)
            #  out = self.bn2(out)

            return out

        elif cfg.filter_mode not in [2,4,8]:
            raise NotImplementedError
        else:
            for i in range(0, self.planes, cfg.filter_mode):
                if i == 0:
                    out = self.conv1(x[:, i:i+cfg.filter_mode,:,:])
                else:
                    temp_out = self.conv1(x[:,i:i+cfg.filter_mode,:,:])
                    out = torch.cat(([out, temp_out]), dim=1)

            return out






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

