from abstract_network import AbstractNet
from torch import nn


class LSTMNet(AbstractNet):
    def __init__(self,
                 input_dimensions,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 nb_output=2,
                 use_conv=False,
                 bidirectional=True,
                 cell_type='lstm',
                 dropout=0.5):

        super(LSTMNet, self).__init__()

        self.use_conv = use_conv
        self.cell_type = cell_type
        if cell_type == 'lstm':
            self.inner_model = nn.LSTM(input_dimensions, hidden_size,
                                       dropout=dropout, num_layers=num_layers,
                                       batch_first=batch_first,
                                       bidirectional=bidirectional)

        elif cell_type == 'gru':
            self.inner_model = nn.GRU(input_dimensions, hidden_size,
                                      dropout=dropout, num_layers=num_layers,
                                      batch_first=batch_first,
                                      bidirectional=bidirectional)
        elif cell_type == 'rnn':
            self.inner_model = nn.RNN(input_dimensions, hidden_size,
                                      num_layers=num_layers,
                                      batch_first=batch_first, dropout=dropout,
                                      bidirectional=bidirectional)

        mult = 2 if bidirectional else 1

        self.output_model = nn.RNN(hidden_size * mult, nb_output,
                                       num_layers=1,
                                       batch_first=batch_first,
                                       nonlinearity='tanh')

    def forward(self, x):
        lstm_out = self.inner_model(x)[0]
        out = self.output_model(lstm_out)
        return out

