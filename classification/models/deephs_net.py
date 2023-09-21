import torch
from torch import nn

class DeepHSNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(DeepHSNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        kernel_count = 3
        self.kernel_count = kernel_count

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_channels * kernel_count, kernel_size=3, padding=1, groups=self.num_channels),
            nn.Conv2d(self.num_channels * kernel_count, 25, kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(25),
            nn.Conv2d(25, 25 * kernel_count, kernel_size=3, padding=1, groups=25),
            nn.Conv2d(25 * kernel_count, 30, kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 30 * kernel_count, kernel_size=3, padding=1, groups=30),
            nn.Conv2d(30 * kernel_count, 50, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(50),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(50),
            nn.Linear(50, self.num_classes)
        )

    def forward(self, x, meta_data=None):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def init_params(self, layers=None, BN=True, seed=0):
        '''Init layer parameters.'''
        #torch.manual_seed(seed)

        modules = self.modules() if layers is None else layers
        for m in modules:
            print('(Re)init params for {}'.format(m))
            if isinstance(m, nn.Conv2d):
                torch.manual_seed(seed)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif BN and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.manual_seed(seed)
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset(self, seed=0):
        self.init_params(seed=seed)

    def reset_first_layer(self, num_channels, seed=0):
        print('Reset first layer -> {} wavelengths/channels'.format(num_channels))
        self.num_channels = num_channels
        self.conv = nn.Sequential(*([nn.Conv2d(self.num_channels, self.num_channels * self.kernel_count, kernel_size=3, padding=1, groups=self.num_channels),
                                     nn.Conv2d(self.num_channels * self.kernel_count, 25, kernel_size=1)] + list(self.conv.children())[2:]))
        self.init_params(layers=nn.Sequential(*list(self.conv.children())[:2]), BN=False, seed=seed)

    def reset_head(self, seed=0, num_classes=None):
        print('Reset head')
        if num_classes is None:
            self.init_params(layers=self.fc, BN=False, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(50, self.num_classes)])
            self.init_params(layers=self.fc, BN=False, seed=seed)

if __name__ == '__main__':
    model = DeepHSNet(num_channels=200, num_classes=3)
    model.init_params()
    model.reset()
    model.reset_first_layer(num_channels=150)
    model.reset_head(num_classes=2)