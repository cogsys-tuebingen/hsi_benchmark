from torch import nn


class MLP(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(MLP, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(self.num_channels, int(2 / 3 * self.num_channels)),
            nn.ReLU(),
            nn.Linear(int(2 / 3 * self.num_channels), self.num_classes),
            # nn.Softmax()
        )

    def forward(self, x, meta_data=None):
        x = x[:, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel

        x = self.fc(x)

        return x
