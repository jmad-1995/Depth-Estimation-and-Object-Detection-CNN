import torch
import torch.nn as nn
import torchvision.models as models


class MultiPurposeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = MobileNetV2()
        self.depth = DepthBranch()

    def forward(self, x):

        p1, p2, p3, p4, p5 = self.backbone(x)

        print(p1.shape)
        print(p2.shape)
        print(p3.shape)
        print(p4.shape)
        print(p5.shape)


class MobileNetV2(nn.Module):

    """Input: 480x640 (4:3), and has height and width divisible by 32"""

    def __init__(self, pretrained=True):
        super().__init__()

        self.mobilenetv2 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][:-5])
        self.c1 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][:2])
        self.c2 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][2:4])
        self.c3 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][4:7])
        self.c4 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][7:14])
        self.c5 = nn.Sequential(*list(models.mobilenet_v2(pretrained=pretrained).children())[0][14:18])

    def forward(self, x):
        p1 = self.c1(x)
        p2 = self.c2(p1)
        p3 = self.c3(p2)
        p4 = self.c4(p3)
        p5 = self.c5(p4)
        return p1, p2, p3, p4, p5


class DepthBranch(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


class ObjectBranch(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Frame rate testing
    zeros = torch.zeros((1, 3, 480, 640)).to(device)

    #
    model = MultiPurposeCNN().to(device)

    #
    _ = model(zeros)

