import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_input_channels = 4
        self.cnn_output_channels = 64
        self.rnn_seq_len = 150        

        # CNN
        pool_sizes = 8, 8, 4
        self.layers = []
        for pool_size in pool_sizes:
            self.layers.append(self._CNN_make_layer(pool_size))
        self.CNN = nn.Sequential(*self.layers)

        # RNN
        rnn_input_shape = 256
        self.GRU = nn.GRU(rnn_input_shape, self.rnn_seq_len, num_layers=2, batch_first=True, bidirectional=True)

        # SED
        num_cls = 11
        self. SED_fc = nn.Linear(2*self.rnn_seq_len, num_cls)

        # DOA
        # each azi, ele have 11 classes 
        # so total classes is 22
        num_cls = 22
        self. DOA_fc = nn.Linear(2*self.rnn_seq_len, num_cls)
        

    def _CNN_make_layer(self, pool_size):
        layer = nn.Sequential(
            nn.Conv2d(self.cnn_input_channels, self.cnn_output_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(self.cnn_output_channels),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size)),
            nn.Dropout()
        )
        self.cnn_input_channels = self.cnn_output_channels
        return layer

    def forward(self, x):
        x = self.CNN(x)
        # for make tensor's shape (batch_size, 3000, 64, 4)
        x = x.permute(0, 2, 1, 3)
        # (20, 150, 256)
        x = x.reshape(20, 150, -1)
        # GRU(x)[0].shape is (20, 150, 300)
        # 300 = self.rnn_seq_len * 2 (bidirectional)
        x = self.GRU(x)[0]
        sed = TimeDistributed(self.SED_fc)(x).reshape(1, 3000, -1)
        doa = TimeDistributed(self.DOA_fc)(x).reshape(1, 3000, -1)
        return sed, doa


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y
        