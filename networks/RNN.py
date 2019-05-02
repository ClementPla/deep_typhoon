from .abstract_network import AbstractNet
from torch import nn
from torch.nn.parameter import Parameter
import torch
from .. import functional as F

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
                 non_linearity='tanh', impute_data = False):

        super(LSTMNet, self).__init__(checkpoint, gpu)

        self.cell_type = cell_type
        self.impute_data = impute_data

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

        if self.impute_data:
            self.imputation_activation = nn.ReLu()
            self.imputation_w = Parameter(torch.Tensor(input_dimensions))
            self.imputation_b = Parameter(torch.Tensor(input_dimensions))

    def forward(self, x):
        lstm_out = self.inner_model(x)[0]
        out = self.output_model(lstm_out)[0]
        return out

    def get_imputation_weight(self, l):
        lin = F.linear(l, torch.diag(self.imputation_w), self.imputation_b)
        return torch.exp(-1*self.imputation_activation(lin))




