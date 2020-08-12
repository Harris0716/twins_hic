import torch.nn as nn
import torch
import torch.nn.functional as F

#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(120, 20),
            nn.ReLU(True),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2

class SLeNet(nn.Module):
    def __init__(self):
        super(SLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(16*19*19, 120),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(True),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2

class SAlexNet(nn.Module):
    def __init__(self):
        super(SAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 1 * 1), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=83),
        )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #out = self.distance(out1, out2)
        return out1, out2