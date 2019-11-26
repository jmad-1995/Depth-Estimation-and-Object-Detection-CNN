import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from skimage.transform import resize


class MultiPurposeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = MobileNetV2()
        self.depth = DepthBranch()
        self.ssd = ObjectBranch()

        # Parameter count
        total = sum(p.numel() for p in self.parameters())
        mobilenetv2 = sum(p.numel() for p in self.backbone.parameters())
        depth = sum(p.numel() for p in self.depth.parameters())
        _object = sum(p.numel() for p in self.ssd.parameters())
        print('\n\n')
        print('# ' * 50)
        print('Multi-purpose CNN initialized with: {:.3e} total parameters'.format(total))
        print('MobileNetV2: {:.3e} total parameters'.format(mobilenetv2))
        print('Depth Estimation Branch: {:.3e} total parameters'.format(depth))
        print('Object Detection Branch: {:.3e} total parameters'.format(_object))
        print('# ' * 50)
        print('\n\n')

    def forward(self, x):

        feature_maps = self.backbone(x)
        depth = self.depth(feature_maps)
        detections = self.ssd(feature_maps)

        return {'depths': depth, 'objects': detections}

    def predict(self, image, device='cpu'):
        self.eval()

        # Convert to float
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.

        # Padding
        aspect_ratio = 1280 / 384
        if image.shape[1] / image.shape[0] > aspect_ratio:
            pad_x = 0
            pad_y = int(image.shape[1] / aspect_ratio - image.shape[0])
        else:
            pad_x = int(image.shape[0] * aspect_ratio - image.shape[1])
            pad_y = 0
        padded = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), constant_values=1e-3,
                        mode='constant')

        # Resize
        resized = resize(padded, (384, 1280), mode='constant', cval=1e-3, anti_aliasing=False).astype(np.float32)

        # Predict on image plus mirror image
        image_flipped = np.flip(resized, axis=1)
        image_tensor = torch.from_numpy(np.stack([resized, image_flipped], axis=0).copy()).permute(0, 3, 1, 2)

        # Forward pass + average of two predictions
        predictions = self.forward(image_tensor.to(device))
        depth = predictions['depths'].squeeze()
        depth = torch.mean(torch.stack([depth[0], torch.flip(depth[1], dims=[1])]), dim=0)

        # Post-process
        depth = depth.detach().cpu().numpy()
        depth = resize(depth, resized.shape[:2], mode='constant', cval=1e-3, anti_aliasing=False)

        # Crop
        x, y = (int(pad_x * resized.shape[1] / padded.shape[1]),
                int(pad_y * resized.shape[0] / padded.shape[0]))

        # print("image shape: ", image.shape)
        # print('resized shape: ', resized.shape)
        # print("padded shape: ", padded.shape)
        # print("pad x", pad_x)
        # print("pad y", pad_y)

        depth = depth[:384-y, :1280-x]
        resized = resized[:384-y, :1280-x, :]

        return resized, depth


class MobileNetV2(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

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
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 24, 4, 2, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(48, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1, groups=1, bias=False))

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


class DeepResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:3])  # 240x320x64
        self.c2 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[3:5])  # 120x160x256
        self.c3 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[5:6])  # 60x80x512
        self.c4 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[6:7])  # 30x40x1024
        self.c5 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[7:8])  # 15x30x2048

        self.up5 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 2, 2, 0), nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 2, 2, 0), nn.BatchNorm2d(128), nn.ReLU())
        self.depth = nn.Conv2d(128, 1, 3, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)  # 240x320x64
        c2 = self.c2(c1)  # 120x160x256
        c3 = self.c3(c2)  # 60x80x512
        c4 = self.c4(c3)  # 30x40x1024
        c5 = self.c5(c4)  # 15x30x2048

        up4 = self.up5(c5)  # 30x40x1024
        up3 = self.up4(torch.cat([up4, c4], dim=1))  # 60x80x512
        up2 = self.up3(torch.cat([up3, c3], dim=1))  # 120x160x256
        up1 = self.up2(torch.cat([up2, c2], dim=1))  # 240x320x64

        depth = self.depth(up1)

        return {'depths': depth}


if __name__ == '__main__':

    zeros = torch.zeros((1, 3, 384, 1280))
    model = MultiPurposeCNN()

    o = model(zeros)
    d = o['depths']
    print(d.shape)