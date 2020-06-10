import torch
from torch import nn


class Pnet(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 10, 3),
                         nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Conv2d(10, 16, 3),
                         nn.ReLU(),
                         nn.Conv2d(16, 32, 3),
                         nn.ReLU(),
                         nn.Conv2d(32, 5, 1)
                         )


class Rnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(28, 48, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 2),
            nn.ReLU(),
        )
        self.ouput_layer = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.reshape(-1, 3 * 3 * 64)
        return self.ouput_layer(h)


class Onet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.ouputlayer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.reshape(-1, 3 * 3 * 128)
        return self.ouputlayer(h)


if __name__ == "__main__":
    # pnet = Pnet()
    # y = pnet(torch.randn(1, 3, 12, 12))
    # print(y.shape)
    rnet = Rnet()
    y = rnet(torch.randn(1, 3, 24, 24))
    print(y.shape)
    # onet = Onet()
    # y = onet(torch.randn(1, 3, 48, 48))
    # print(y.shape)
