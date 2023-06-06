class CNN(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, drop_out):
        super().__init__()
        prev_dim = num_channels
        k = kernel_size
        layers = [nn.Conv1d(1, prev_dim, kernel_size=k, padding='same'), nn.ReLU(), nn.Dropout(p=drop_out)]

        for _ in range(1, num_layers):
            layers.append(nn.Conv1d(prev_dim, num_channels, kernel_size=k, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))

        layers.append(nn.Conv1d(prev_dim, 1, kernel_size=k, padding='same'))
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return self.m(x)