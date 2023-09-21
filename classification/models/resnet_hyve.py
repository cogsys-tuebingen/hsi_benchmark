from torch.hub import load_state_dict_from_url

from classification.models.utils.hyve_conv.hyve_convolution import HyVEConv
from classification.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from dataloader.basic_dataloader import get_channel_wavelengths
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ResnetHyve(ResNet):
    def __init__(self, block, layers, wavelength_range, num_of_wrois, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, num_channels=3):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer, num_channels)
        self.wavelength_range = wavelength_range
        self.conv1 = HyVEConv(
            num_of_wrois=num_of_wrois,
            wavelength_range=wavelength_range,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)

        #self.fc = nn.Sequential(nn.Linear(512 * self.block.expansion, 100), nn.Linear(100, 50), nn.Linear(50, self.num_classes))
        self.init_params()

    def get_backbone(self):
        return nn.Sequential(*list(self.children())[:-1])

    def get_head(self):
        return nn.Sequential(self.fc)

    def forward(self, x, meta_data=None):
        channel_wavelengths = get_channel_wavelengths(meta_data).type_as(x)
        x = self.conv1(x, channel_wavelengths=channel_wavelengths)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def init_params(self, layers=None, Norm=True, seed=0):
        #torch.manual_seed(seed)

        modules = self.modules() if layers is None else layers
        # if layers is None:
        #     modules = self.modules()
        #
        # else:
        #     modules = [layers] if isinstance(layers, nn.Linear) else layers

        for m in modules:
            #print('(Re)init params for {}'.format(m))
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                torch.manual_seed(seed)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, HyVEConv):
                m.initialize(seed=seed)
            if Norm and isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.zero_init_residual:
            for m in modules:
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def reset(self, seed=0):
        self.init_params(seed=seed)

    def reset_head(self, seed=0, num_classes=None, BN=False):
        print('Reset head')
        if num_classes is None:
            self.init_params(layers=nn.Sequential(self.fc), Norm=BN, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            self.init_params(layers=nn.Sequential(self.fc), Norm=BN, seed=seed)

    def reset_hyve_layer_gaussian_distributions(self, wavelength_range: Tuple[float, float], seed=0):
        torch.manual_seed(seed)
        self.conv1.initialize_gaussian_distribution(wavelength_range)

    def reset_hyve_layer(self, seed=0):
        torch.manual_seed(seed)
        self.conv1.initialize()

    def reset_first_layer(self, wavelength_range, seed=0):
        print('Reset first layer -> wavelength range {}'.format(wavelength_range))
        self.wavelength_range = wavelength_range
        self.reset_hyve_layer_gaussian_distributions(wavelength_range=wavelength_range, seed=seed)

    def freeze_backbone(self):
        print('Freeze model backbone params')
        for n, p in self.named_parameters():
            if not (n.startswith("fc.") or n.startswith("conv1.")):
                #print('Freeze parameter {}'.format(n))
                p.requires_grad = False

    def unfreeze_backbone(self):
        print('Unfreeze model backbone params')
        for n, p in self.named_parameters():
            if not (n.startswith("fc.") or n.startswith("conv1.")):
                # print('Unfreeze parameter {}'.format(n))
                p.requires_grad = True


def _resnet_hyve(arch, block, layers, pretrained, progress, num_channels, num_classes, wavelength_range, num_of_wrois, seed=0, BN=False, **kwargs):
    if pretrained:
        model = ResnetHyve(block, layers, num_classes=1000, wavelength_range=wavelength_range, num_of_wrois=num_of_wrois, **kwargs)
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False) # ignore 1st conv = hyve conv layer (wavelengths)
        model.reset_head(num_classes=num_classes, seed=seed, BN=BN)
    else:
        model = ResnetHyve(block, layers, num_classes=num_classes, wavelength_range=wavelength_range, num_of_wrois=num_of_wrois, **kwargs)
    return model


def ResNetHyve18(num_channels, num_classes, wavelength_range, num_of_wrois):
    return _resnet_hyve('resnet18', BasicBlock, [2, 2, 2, 2],
                        False, True,
                        num_channels=num_channels, num_classes=num_classes,
                        wavelength_range=wavelength_range, num_of_wrois=num_of_wrois)

def ResNetHyve152(num_channels, num_classes, wavelength_range, num_of_wrois):
    return _resnet_hyve('resnet152', Bottleneck, [3, 8, 36, 3],
                        False, True,
                        num_channels=num_channels, num_classes=num_classes,
                        wavelength_range=wavelength_range, num_of_wrois=num_of_wrois)


if __name__ == '__main__':
    # model = ResNetHyve18(num_channels=200, num_classes=3, wavelength_range=(400, 1000), num_of_wrois=5)
    # model.init_params(seed=0)
    # for n, p in model.named_parameters():
    #     if n.__contains__('conv1.kernel_weights_individual'):
    #         print(n, p)
    # model.reset(seed=0)
    # for n, p in model.named_parameters():
    #     if n.__contains__('conv1.kernel_weights_individual'):
    #         print(n, p)

    print('Model 1:')
    model = ResNetHyve18(num_channels=200, num_classes=3, wavelength_range=(400, 1000), num_of_wrois=5)
    for n, p in model.fc.named_parameters():
        print(n, p)
    model.init_params(seed=3)
    for n, p in model.fc.named_parameters():
        print(n, p)
    model.reset_head(num_classes=2, seed=5)
    for n, p in model.fc.named_parameters():
        print(n, p)

    print('\n Model 2:')
    model2 = ResNetHyve18(num_channels=200, num_classes=3, wavelength_range=(400, 1000), num_of_wrois=5)
    for n, p in model2.fc.named_parameters():
        print(n, p)
    model2.init_params(seed=3)
    for n, p in model2.fc.named_parameters():
        print(n, p)
    model2.reset_head(num_classes=2, seed=5)
    for n, p in model2.fc.named_parameters():
        print(n, p)
