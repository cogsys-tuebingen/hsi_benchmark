from torch import nn


class SimpleRNN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int,
                 mode: str = 'vanilla'):
        super(SimpleRNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.mode = mode
        if self.mode == 'vanilla':
            self.recurr = nn.RNN(self.num_channels, 64, num_layers=2, batch_first=True)
        elif self.mode == 'lstm':
            self.recurr = nn.LSTM(self.num_channels, 64, num_layers=2, batch_first=True)
        elif self.mode == 'gru':
            self.recurr = nn.GRU(self.num_channels, 64, num_layers=2, batch_first=True)

        # self.bn = nn.BatchNorm1d(64)

        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x, meta_data=None):
        x = x[:, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel

        x, _ = self.recurr(x)
        # x = self.bn(x)
        x = self.fc(x)

        return x