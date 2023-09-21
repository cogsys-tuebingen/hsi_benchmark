import torch.nn as nn
import torchvision.transforms as T

class HybridSN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, image_size: int = 63):
        super(HybridSN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.image_size = image_size

        flattened_dim = (self.num_channels - (7 - 1) - (5 - 1) - (3 - 1)) * 32

        backbone_out_dim = (self.image_size - 4 * (3 - 1)) * (self.image_size - 4 * (3 - 1)) * 64

        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Conv2d(flattened_dim, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x, meta_data=None):
        # if not (x.shape[2] == 63 and x.shape[3] == 63):
        #     x = T.functional.resize(x, (63, 63))

        x = self.conv(x.unsqueeze(1))
        x = self.fc(x)

        return x


if __name__ == '__main__':
    from classification.model_factory import get_model_size
    m = HybridSN(30, 10, 25)

    print(get_model_size("hybridsn"))

