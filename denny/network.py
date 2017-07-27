import torch.nn as nn
import torch.nn.functional as F

class BvlcAlexNet(nn.Module):
    def __init__(self):
        super(BvlcAlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*5*5,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,5)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x


class Cifar10FullNet(nn.Module):
    def __init__(self):
        super(Cifar10FullNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Sigmoid(),
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=2),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3,stride=2)
        )
        self.classifier = nn.Linear(64*27*27,5)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x

