import torch.nn as nn
import torch
from typing import Optional, Tuple

from classification.models.utils.hyve_conv.hyve_convolution import HyVEConv
from dataloader.basic_dataloader import get_channel_wavelengths


class DeepHSNet_with_HyVEConv(nn.Module):
    def __init__(self, wavelength_range, num_of_wrois, enable_extension=True,
                 num_classes=3, stop_gaussian_gradient=False, num_channels=None):
        super(DeepHSNet_with_HyVEConv, self).__init__()
        self.bands = num_channels
        self.num_classes=num_classes
        self.hidden_layers = [25, 30, 50]

        self.wavelength_range = wavelength_range
        self.gauss_variance_factor = (wavelength_range[1] - wavelength_range[0])

        # FIXME dynamic channel num
        self.learnable_linear_interpolation = None

        self.hyve_conv = HyVEConv(num_of_wrois=num_of_wrois,
                                           wavelength_range=self.wavelength_range,
                                           out_channels=self.hidden_layers[0],
                                           kernel_size=7,
                                           enable_extension=enable_extension,
                                           stop_gaussian_gradient=stop_gaussian_gradient
                                           )

        kernel_count = 3

        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.Conv2d(self.hidden_layers[0], self.hidden_layers[0] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[0]),
            nn.Conv2d(self.hidden_layers[0] * kernel_count,
                      self.hidden_layers[1], kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(self.hidden_layers[1]),
            nn.Conv2d(self.hidden_layers[1], self.hidden_layers[1] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[1]),
            nn.Conv2d(self.hidden_layers[1] * kernel_count,
                      self.hidden_layers[2], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_layers[2]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[2]),
            nn.Linear(self.hidden_layers[2], num_classes),
        )

        self.init_params()

    def get_hyve_conv(self):
        return self.hyve_conv

    def get_backbone(self):
        return self.conv

    def get_head(self):
        return self.fc

    def forward(self, x, meta_data=None):
        assert meta_data is not None
        channel_wavelengths = get_channel_wavelengths(meta_data).type_as(x)

        out = self.hyve_conv(x, channel_wavelengths=channel_wavelengths)
        out = self.conv(out)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out

    def init_params(self, layers=None, BN=True, seed=0):
        '''Init layer parameters.'''
        #torch.manual_seed(seed)

        modules = self.modules() if layers is None else layers
        for m in modules:
            #print('(Re)init params for {}'.format(m))
            if isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                torch.manual_seed(seed)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, HyVEConv):
                m.initialize(seed=seed)
            if BN and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset(self, seed=0):
        self.init_params(seed=seed)

    def reset_head(self, seed=0, num_classes=None, BN=False):
        print('Reset head')
        if num_classes is None:
            self.init_params(layers=self.fc, BN=BN, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(self.hidden_layers[-1], self.num_classes)])
            self.init_params(layers=self.fc, BN=BN, seed=seed)

    def reset_hyve_layer_gaussian_distributions(self, wavelength_range: Tuple[float, float], seed=0):
        torch.manual_seed(seed)
        self.hyve_conv.initialize_gaussian_distribution(wavelength_range=wavelength_range)

    def reset_hyve_layer(self, seed=0):
        torch.manual_seed(seed)
        self.hyve_conv.initialize()

    def reset_first_layer(self, wavelength_range, seed=0):
        print('Reset first layer -> wavelength range {}'.format(wavelength_range))
        self.wavelength_range = wavelength_range
        self.reset_hyve_layer_gaussian_distributions(wavelength_range=self.wavelength_range, seed=seed)

    def freeze_backbone(self):
        print('Freeze model backbone params')
        for n, p in self.conv.named_parameters():
            #print('Freeze parameter {}'.format(n))
            p.requires_grad = False

    def unfreeze_backbone(self):
        print('Unfreeze model backbone params')
        for n, p in self.conv.named_parameters():
            #print('Unfreeze parameter {}'.format(n))
            p.requires_grad = True



class Larger_DeepHSNet_with_HyVEConv(DeepHSNet_with_HyVEConv):
    def __init__(self, num_channels, wavelength_range, num_of_wrois, enable_extension=True,
                 num_classes=3, stop_gaussian_gradient=False):
        super(DeepHSNet_with_HyVEConv, self).__init__()
        self.bands = num_channels
        #self.hidden_layers = [25, 30, 50, 100, 100, 50] # add 3 hidden layers
        self.hidden_layers = [50, 60, 100, 200, 500, 200, 100, 50] # add 5 larger hidden layers

        self.wavelength_range = wavelength_range
        self.gauss_variance_factor = (wavelength_range[1] - wavelength_range[0])

        # FIXME dynamic channel num
        self.learnable_linear_interpolation = None

        self.hyve_conv = HyVEConv(num_of_wrois=num_of_wrois,
                                           wavelength_range=self.wavelength_range,
                                           out_channels=self.hidden_layers[0],
                                           kernel_size=7,
                                           enable_extension=enable_extension,
                                           stop_gaussian_gradient=stop_gaussian_gradient
                                           )

        kernel_count = 3

        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(self.hidden_layers[0]),
            nn.Conv2d(self.hidden_layers[0], self.hidden_layers[0] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[0]),
            nn.Conv2d(self.hidden_layers[0] * kernel_count,
                      self.hidden_layers[1], kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(self.hidden_layers[1]),
            nn.Conv2d(self.hidden_layers[1], self.hidden_layers[1] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[1]),
            nn.Conv2d(self.hidden_layers[1] * kernel_count,
                      self.hidden_layers[2], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_layers[2]),
            # add. layer 1
            nn.Conv2d(self.hidden_layers[2], self.hidden_layers[2] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[2]),
            nn.Conv2d(self.hidden_layers[2] * kernel_count,
                      self.hidden_layers[3], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_layers[3]),
            # add layer 2
            nn.Conv2d(self.hidden_layers[3], self.hidden_layers[3] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[3]),
            nn.Conv2d(self.hidden_layers[3] * kernel_count,
                      self.hidden_layers[4], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_layers[4]),
            # add layer 3
            nn.Conv2d(self.hidden_layers[4], self.hidden_layers[4] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[4]),
            nn.Conv2d(self.hidden_layers[4] * kernel_count,
                      self.hidden_layers[5], kernel_size=1),
            nn.ReLU(True),
            # add layer 4
            nn.Conv2d(self.hidden_layers[5], self.hidden_layers[5] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[5]),
            nn.Conv2d(self.hidden_layers[5] * kernel_count,
                      self.hidden_layers[6], kernel_size=1),
            nn.ReLU(True),
            # add layer 5
            nn.Conv2d(self.hidden_layers[6], self.hidden_layers[6] * kernel_count,
                      kernel_size=3, padding=1, groups=self.hidden_layers[6]),
            nn.Conv2d(self.hidden_layers[6] * kernel_count,
                      self.hidden_layers[7], kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.hidden_layers[-1]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(self.hidden_layers[-1]),
            nn.Linear(self.hidden_layers[-1], num_classes),
        )

        self.init_params()


if __name__ == '__main__':
    model = DeepHSNet_with_HyVEConv(num_channels=200, num_classes=3, wavelength_range=[400, 1000], num_of_wrois=5)
    for n, p in model.fc.named_parameters():
        print(n, p)
    model.init_params(seed=3)
    for n, p in model.fc.named_parameters():
        print(n, p)
    model.reset_head(num_classes=2, seed=5)
    for n, p in model.fc.named_parameters():
        print(n, p)
    # model.reset_first_layer(wavelength_range=(300,500))
    # for n, p in model.get_backbone().named_parameters():
    #     print(n)

    # model2 = DeepHSNet_with_HyVEConv(num_channels=200, num_classes=3, wavelength_range=[400, 1000], num_of_wrois=5)
    # for n, p in model2.fc.named_parameters():
    #     print(n, p)
    # model2.init_params(seed=3)
    # for n, p in model2.fc.named_parameters():
    #     print(n, p)
    # model2.reset_head(num_classes=2, seed=5)
    # for n, p in model2.fc.named_parameters():
    #     print(n, p)
