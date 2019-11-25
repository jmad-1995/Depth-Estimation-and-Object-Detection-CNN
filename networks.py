import torch
import torch.nn as nn
import torchvision.models as models


class MultiPurposeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = MobileNetV2()
        self.depth = DepthBranch()
        self.ssd = ObjectBranch()

        # Parameter count
        total = sum([p.numel() for p in self.parameters()])
        mobilenetv2 = sum([p.numel() for p in self.backbone.parameters()])
        depth = sum([p.numel() for p in self.depth.parameters()])
        object = sum([p.numel() for p in self.ssd.parameters()])
        print('Multi-purpose CNN initialized with: {:.3e total parameters}'.format(total))
        print('MobileNetV2: {:.3e total parameters}'.format(mobilenetv2))
        print('Depth Estimation Branch: {:.3e total parameters}'.format(depth))
        print('Object Detection Branch: {:.3e total parameters}'.format(object))

    def forward(self, x):

        feature_maps = self.backbone(x)

        # p1, p2, p3, p4, p5 = feature_maps
        # # print("P1: ", p1.shape)
        # # print("P2: ", p2.shape)
        # # print("P3: ", p3.shape)
        # # print("P4: ", p4.shape)
        # # print("P5: ", p5.shape)

        depth = self.depth(feature_maps)
        detections = self.ssd(feature_maps)

        return depth, detections


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

        # Up 5
        self.up4 = nn.Sequential(nn.ConvTranspose2d(320, 96, 4, 2, 1, bias=False), nn.BatchNorm2d(96), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(192, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 24, 2, 2, 0, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(48, 16, 2, 2, 0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(32, 1, 3, padding=1, groups=1, bias=False))

    def forward(self, feature_maps):
        p1, p2, p3, p4, p5 = feature_maps

        up4 = self.up4(p5)
        up3 = self.up3(torch.cat([up4, p4], dim=1))
        up2 = self.up2(torch.cat([up3, p3], dim=1))
        up1 = self.up1(torch.cat([up2, p2], dim=1))
        depth = self.out(torch.cat([up1, p1], dim=1))

        return depth


class ObjectBranch(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_maps):
        p1, p2, p3, p4, p5 = feature_maps
        return None


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Frame rate testing
    zeros = torch.zeros((1, 3, 480, 640)).to(device)

    #
    model = MultiPurposeCNN().to(device)

    #
    d, o = model(zeros)

    print(d.shape)
