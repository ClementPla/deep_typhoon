from .abstract_network import AbstractNet
from torch import nn
from torch.nn.parameter import Parameter
import torch

class LSTMNet(AbstractNet):
    def __init__(self,
                 checkpoint,
                 input_dimensions,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 nb_output=2,
                 bidirectional=True,
                 cell_type='lstm',
                 dropout=0.5,
                 gpu=0,
                 non_linearity='tanh', impute = False):

        super(LSTMNet, self).__init__(checkpoint, gpu)

        self.cell_type = cell_type
        self.impute = impute
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
                                       nonlinearity=non_linearity)

        if self.impute:
            self.imputate_model = nn.RNN(input_dimensions+1, input_dimensions, 2, non_linearity='relu')

    def forward(self, x):
        lstm_out = self.inner_model(x)[0]
        out = self.output_model(lstm_out)[0]
        return out

    def imputation(self, x, m, l):
        b, s = x.size()[:2]
        input_imputation = torch.cat([x.view(b * s, -1), l.view(b * s, -1)], dim=1).view(b, s, -1)
        x_predicted = self.imputate_model(input_imputation)[0]
        return x * m + (1 - m) * x_predicted




