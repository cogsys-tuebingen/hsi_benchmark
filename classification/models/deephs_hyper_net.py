import torch
from torch import nn
import torch

from classification.models.utils.hyve_conv.hyve_convolution import HyVEConv
from dataloader.basic_dataloader import get_channel_wavelengths
from classification.models.deephs_hyve_net import DeepHSNet_with_HyVEConv

class DeepHSHybridNet_with_HyveConv(DeepHSNet_with_HyVEConv):
    def __init__(self, wavelength_range, num_of_wrois, enable_extension=True,
                 num_classes=3, stop_gaussian_gradient=False, num_channels=None):
        super(DeepHSHybridNet_with_HyveConv, self).__init__(
            num_channels=num_channels,
            wavelength_range=wavelength_range,
            num_of_wrois=num_of_wrois,
            enable_extension=enable_extension,
            num_classes=num_classes,
            stop_gaussian_gradient=stop_gaussian_gradient
        )

        out_channels = (self.hidden_layers[0] - (7 - 1)) * 25

        self.conv = nn.Sequential(
            nn.Conv3d(1, 25, kernel_size=(7, 3, 3), padding=0),  # single input channel
            nn.ReLU(True),
            nn.Flatten(start_dim=1, end_dim=2), # flatten channels & wavelengths
            nn.AvgPool2d((4, 4)),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, 30, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d((4, 4)),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 50, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(50),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.shortcut = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(75),
            nn.Linear(75, num_classes),
        )

        self.init_params()

    def forward(self, x, meta_data=None):
        assert meta_data is not None
        channel_wavelengths = get_channel_wavelengths(meta_data).type_as(x)
        x_p = self.hyve_conv(x, channel_wavelengths=channel_wavelengths)
        x = x_p.unsqueeze(1)  # single input channel
        x = self.conv(x)
        x = x.view(x.shape[0], -1)

        x_p = self.shortcut(x_p)
        x_p = x_p.view(x_p.shape[0], -1)

        x = torch.concat((x, x_p), dim=1)
        x = self.fc(x)

        return x

    def init_params(self, layers=None, BN=True, seed=0):
        '''Init layer parameters.'''
        #torch.manual_seed(seed)

        modules = self.modules() if layers is None else layers
        for m in modules:
            print('(Re)init params for {}'.format(m))
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                torch.manual_seed(seed)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, HyVEConv):
                m.initialize(seed=seed)
            elif BN and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_head(self, seed=0, num_classes=None, BN=False):
        print('Reset head')
        if num_classes is None:
            self.init_params(layers=self.fc, BN=BN, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(75, self.num_classes)])
            self.init_params(layers=self.fc, BN=BN, seed=seed)


if __name__ == '__main__':
    model = DeepHSHybridNet_with_HyveConv(wavelength_range=(400, 1000), num_of_wrois=5, num_classes=3)
    # model.reset()
    # model.reset_head(num_classes=2)
    for n, p in model.get_backbone().named_parameters():
        print(n)
    for n, p in model.get_head().named_parameters():
        print(n)
