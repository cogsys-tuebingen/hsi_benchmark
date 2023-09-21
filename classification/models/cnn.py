import torch
from torch import nn
from abc import ABC, abstractmethod


class CNN(nn.Module, ABC):
    def __init__(self, num_channels: int, num_classes: int):
        super(CNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x, meta_data=None):
        pass

    def reset(self, seed=0):
        self.apply(lambda x: self._weights_reset(x, seed))

    def reset_first_layer(self, num_channels):
        print('Reset first layer -> {} wavelengths/channels'.format(num_channels))
        self.num_channels = num_channels
        raise NotImplementedError

    def reset_head(self, seed=0, num_classes=None):
        print('Reset head')
        if num_classes is None:
            self.fc.apply(lambda x: self._weights_reset(x, seed))
        else:
            print('-> {} classes'.format(num_classes))
            self.num_classes = num_classes
            self.fc = nn.Sequential(*list(self.fc.children())[:-1] + [nn.Linear(100, self.num_classes)])

    def _weights_reset(self, layer, seed=0):
        torch.manual_seed(seed)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d) or isinstance(layer, nn.Linear):
            # if hasattr(layer, 'reset_parameters'):
            print('Reset params for layer {}'.format(layer))
            layer.reset_parameters()
            # torch.nn.init.xavier_uniform(layer.weight.data)


class CNN1D(CNN):
    '''
    1D CNN: spectral
    Input: 1 x 1 x num_bands (spectrum of center pixel)
    '''
    def __init__(self, num_channels: int, num_classes: int):
        super(CNN1D, self).__init__(num_channels, num_classes)

        input_shape = (self.num_channels - (24 - 1)) // 5 * 20

        self.conv = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=24),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
        )

    def forward(self, x, meta_data=None):
        x = x[:, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel
        x = x.unsqueeze(dim=1)

        x = self.conv(x)
        x = self.fc(x)

        return x


class CNN2D(CNN):
    '''
    2D CNN: spectral & spatial
    Input: image_size x image_size x num_bands
    '''
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63):
        super(CNN2D, self).__init__(num_channels, num_classes)

        self.image_size = image_size

        input_shape = ((self.image_size - (5 - 1) - (5 - 1)) // 2) ** 2 * 100

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(50, 100, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            # nn.Softmax()
        )

    def forward(self, x, meta_data=None):
        # if not (x.shape[2] == 63 and x.shape[3] == 63):
        #     x = T.functional.resize(x, (63, 63), antialias=False)

        x = self.conv(x)
        x = self.fc(x)

        return x

class CNN2DSpatial(CNN):
    '''
    2D CNN spatial
    Input:image_size x image_size x 1 (e.g. 1st PC / mean)
    '''
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63,
                 mode: str = 'pca'):
        super(CNN2DSpatial, self).__init__(num_channels, num_classes)

        self.mode = mode

        self.num_channels = 1

        self.image_size = image_size

        input_shape = ((self.image_size - (5 - 1) - (5 - 1)) // 2) ** 2 * 100

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(50, 100, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            # nn.Softmax()
        )

    def forward(self, x, meta_data=None):
        # single channel
        if x.shape[1] > 1:
            if self.mode == 'pca':
                x = torch.unsqueeze(x[:, 0, :, :], dim=1) # 1st principal component (after PCA)
            elif self.mode == 'mean':
                x = torch.unsqueeze(x.mean(dim=1), dim=1)  # mean over channels
            else:
                raise NotImplementedError
        assert x.shape[1] == 1, 'Use with 1st PC or mean over channels only'

        # if not (x.shape[2] == 63 and x.shape[3] == 63):
        #     x = T.functional.resize(x, (63, 63), antialias=False)

        x = self.conv(x)
        x = self.fc(x)

        return x

class CNN2DSpectral(CNN):
    '''
    2D CNN spectral
    Input: 1 x 1 x num_bands (center pixel -> dublicated to image_size x image_size x num_bands)
    '''
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63):
        super(CNN2DSpectral, self).__init__(num_channels, num_classes)

        self.image_size = image_size

        input_shape = ((self.image_size - (5 - 1) - (5 - 1)) // 2) ** 2 * 100

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(50, 100, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            # nn.Softmax()
        )

    def forward(self, x, meta_data=None):
        # dublicate center pixel
        b, c, h, w = x.shape
        x = torch.unsqueeze(torch.unsqueeze(x[:, :, h // 2, w // 2], dim=-1), dim=-1).repeat((1, 1, self.image_size, self.image_size))

        x = self.conv(x)
        x = self.fc(x)

        return x


class CNN3D(CNN):
    '''
    3D CNN: patial & spectral
    Input: image_size x image_size x num_bands
    '''
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63):
        super(CNN3D, self).__init__(num_channels, num_classes)

        self.image_size = image_size

        input_shape = ((self.image_size - (5 - 1) - (5 - 1)) // 2) ** 2 * (self.num_channels - (24 - 1) - (16 - 1)) * 64

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(24, 5, 5)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(16, 5, 5)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, self.num_classes),
            # nn.Softmax()
        )

    def forward(self, x, meta_data=None):
        # if not (x.shape[2] == 63 and x.shape[3] == 63):
        #     x = T.functional.resize(x, (63, 63), antialias=False)

        x = torch.unsqueeze(x, dim=1) # add channel dimension

        x = self.conv(x)
        x = self.fc(x)

        return x
