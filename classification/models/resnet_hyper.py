from classification.models.resnet_hyve import _resnet_hyve
from classification.models.resnet import conv3x3
import torch.nn as nn


def conv3x3_3d(in_planes, out_plane_factor, stride=1, groups=1, dilation=1):
    return nn.Sequential(
        nn.Unflatten(1, (1, in_planes)),
        nn.Conv3d(1, stride, kernel_size=3, stride=(1, stride, stride),
                     padding=(1, dilation, dilation), groups=groups, bias=False, dilation=(1, dilation, dilation)),
        # nn.Conv3d(out_plane_factor, stride, kernel_size=3, stride=(1, stride, stride),
        #              padding=(1, dilation, dilation), groups=groups, bias=False, dilation=(1, dilation, dilation)),
        nn.Flatten(start_dim=1, end_dim=2)
    )


class BasicBlockHybrid(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockHybrid, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        out_plane_factor = 5
        self.conv1 = conv3x3_3d(inplanes, out_plane_factor=out_plane_factor, stride=stride)
        # self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def ResNetHyper18(num_channels, num_classes, wavelength_range, num_of_wrois):
    return _resnet_hyve('resnet18', BasicBlockHybrid, [2, 2, 2, 2],
                        False, True,
                        num_channels=num_channels, num_classes=num_classes,
                        wavelength_range=wavelength_range, num_of_wrois=num_of_wrois)


if __name__ == '__main__':
    model = ResNetHyper18(num_channels=200, num_classes=3, wavelength_range=(400, 1000), num_of_wrois=5)
    model.init_params(seed=0)

