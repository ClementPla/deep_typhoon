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
                 impute=False,
                 output_cell='rnn',
                 output_activation='rnn',
                 learn_hidden_state=False):

        super(LSTMNet, self).__init__(checkpoint, gpu)

        self.cell_type = cell_type
        self.impute = impute
        self.output_cell = output_cell
        self.nb_output = nb_output
        self.learn_hidden_state = learn_hidden_state

        if num_layers == 1:
            dropout = 0
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

        self.hidden_dimension = mult*hidden_size

        if self.output_cell == 'rnn':
            self.last_cell = nn.RNN(hidden_size * mult, nb_output,
                                      num_layers=1,
                                      batch_first=batch_first,
                                      nonlinearity=output_activation)
        elif self.output_cell == 'fc':
            self.last_cell = nn.Linear(hidden_size * mult, nb_output)

        if self.impute:
            self.imputate_model = nn.RNN(input_dimensions + 1, input_dimensions // mult, 2, nonlinearity='relu',
                                         bidirectional=bidirectional)

        if self.learn_hidden_state:
            self.h0_i = nn.Parameter(torch.zeros(num_layers*mult,1, hidden_size))
            self.c0_i = nn.Parameter(torch.zeros(num_layers*mult,1, hidden_size))

            if self.output_cell == 'rnn':
                self.h0_o = nn.Parameter(torch.zeros(1,1, nb_output))
                self.c0_o = nn.Parameter(torch.zeros(1, 1, nb_output))

    def forward(self, x):
        if self.learn_hidden_state:
            b = torch.max(x.batch_sizes)
            lstm_out, _ = self.inner_model(x, (self.h0_i.repeat(1, b, 1), self.c0_i.repeat(1, b, 1)))
        else:
            lstm_out, _ = self.inner_model(x)

        if self.output_cell == 'direct':
            return lstm_out

        elif self.output_cell == 'fc':
            time_step = lstm_out.size(1)
            l_out = lstm_out.view(-1, self.hidden_dimension)
            outs = self.last_cell(l_out).view(-1, time_step, self.nb_output)
            return outs

        elif self.output_cell == 'rnn':
            if self.learn_hidden_state:
                out, _ = self.last_cell(lstm_out, self.(self.h0_o.repeat(1, b, 1), self.c0_o.repeat(1, b, 1)))
            else:
                out, _ = self.last_cell(lstm_out)
            return out

        else:
            raise ValueError('Not expected cell type', self.output_cell)

    def imputation(self, x, m, l):
        b, s = x.size()[:2]
        x_flat = torch.flatten(x, 0, 1)
        l = torch.flatten(l, 0, 1)
        cat_tensor = torch.cat((x_flat, l), 1)
        input_imputation = cat_tensor.view((b, s, -1))
        x_predicted = self.imputate_model(input_imputation)[0]
        return x * m + (1 - m) * x_predicted
