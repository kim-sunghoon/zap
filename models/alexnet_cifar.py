import torch
import torch.nn as nn
from PredNet import ZAP, PredNet


class AlexNet(PredNet):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.do1 = nn.Dropout()
        self.fc1 = nn.Linear(4096, 2048)
        self.do2 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)

        self.pred1 = ZAP(192)
        self.pred2 = ZAP(384)
        self.pred3 = ZAP(256)
        self.pred4 = ZAP(256)

        self.pred_layers = [self.pred1, self.pred2, self.pred3, self.pred4]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.pred1(x)

        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.pred2(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.pred3(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.pred4(x)

        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)

        x = self.do1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def alexnet(num_classes=10, **kwargs):
    return AlexNet(num_classes=num_classes)


def alexnet_cifar10(**kwargs):
    return AlexNet(num_classes=10)


def alexnet_cifar100(**kwargs):
    return AlexNet(num_classes=100)
