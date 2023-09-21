from torch import nn
from sample_transforms.emp import ExtendedMorphologicalProfiles

class EMPCNN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63):
        super(EMPCNN, self).__init__()

        assert num_channels == 3

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.image_size = image_size

        input_shape = ((self.image_size - (5 - 1) - (5 - 1)) // 2) ** 2 * 100

        self.emp = ExtendedMorphologicalProfiles(data=None)
        self.conv = nn.Sequential(
            nn.Conv2d((4*2+1) * self.num_channels, 50, kernel_size=(5, 5)),
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
        )

    def forward(self, x, meta_data=None):

        x = self.emp(x).type_as(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
