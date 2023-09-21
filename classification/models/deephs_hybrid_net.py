import torch
from torch import nn

from classification.models.deephs_net import DeepHSNet


class DeepHSHybridNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(DeepHSHybridNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        out_channels = (self.num_channels - (7 - 1)) * 25

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

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(50),
            nn.Linear(50, self.num_classes)
        )

        self.init_params()

    def forward(self, x, meta_data=None):
        x = x.unsqueeze(1)  # single input channel
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def init_params(self, layers=None, BN=True, seed=0):
        '''Init layer parameters.'''
        #torch.manual_seed(seed)

        modules = self.modules() if layers is None else layers
        for m in modules:
            #print('(Re)init params for {}'.format(m))
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
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
        raise NotImplementedError

    def reset_head(self, seed=0, num_classes=None):
        print('Reset head')
        if num_classes is None:
            self.init_params(self.fc, BN=False, seed=seed)
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(50, self.num_classes)])
            self.init_params(self.fc, BN=False, seed=seed)

    def freeze_backbone(self):
        for n, p in self.conv.named_parameters():
            print('Freeze parameter {}'.format(n))
            p.requires_grad = False

    def unfreeze_backbone(self):
        for n, p in self.conv.named_parameters():
            print('Unfreeze parameter {}'.format(n))
            p.requires_grad = True


if __name__ == '__main__':
    model = DeepHSHybridNet(num_channels=200, num_classes=3)
    model.reset()
    #model.reset_first_layer(num_channels=150)
    model.reset_head(num_classes=2)
