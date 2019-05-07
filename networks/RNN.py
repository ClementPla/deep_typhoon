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
            self.output_cell = nn.RNN(hidden_size * mult, nb_output,
                                      num_layers=1,
                                      batch_first=batch_first,
                                      nonlinearity=output_activation)
        elif self.output_cell == 'fc':
            self.output_cell = nn.Linear(hidden_size * mult, nb_output)

        if self.impute:
            self.imputate_model = nn.RNN(input_dimensions + 1, input_dimensions // mult, 2, nonlinearity='relu',
                                         bidirectional=bidirectional)

        if self.learn_hidden_state:
            self.input_h = nn.Parameter(torch.zeros(num_layers*mult,1, hidden_size))
            self.input_c = nn.Parameter(torch.zeros(num_layers*mult,1, hidden_size))

            self.hidden_i = (self.input_h, self.input_c)

            if self.output_cell == 'rnn':
                self.output_h = nn.Parameter(torch.zeros(1,1, nb_output))
                self.output_c = nn.Parameter(torch.zeros(1, 1, nb_output))
                self.hidden_o = (self.output_h, self.output_c)

    def forward(self, x, seq_length=None):
        if self.learn_hidden_state:
            self.hidden_i = tuple([_.repeat(1, torch.max(x.batch_sizes), 1) for _ in self.hidden_i])
            if self.output_cell == 'rnn':
                self.hidden_o = tuple([_.repeat(1, torch.max(x.batch_sizes), 1) for _ in self.hidden_o])

        print(self.hidden_i)
        if self.learn_hidden_state:
            lstm_out = self.inner_model(x, self.hidden_i)[0]
        else:
            lstm_out = self.inner_model(x)[0]

        if self.output_cell == 'direct':
            return lstm_out

        elif self.output_cell == 'fc':
            time_step = lstm_out.size(1)
            l_out = lstm_out.view(-1, self.hidden_dimension)
            outs = self.output_cell(l_out).view(-1, time_step, self.nb_output)
            return outs

        elif self.output_cell == 'rnn':
            if self.learn_hidden_state:
                out = self.output_cell(lstm_out, self.hidden_o)[0]
            else:
                out = self.output_cell(lstm_out)[0]
            return out

    def imputation(self, x, m, l):
        b, s = x.size()[:2]
        x_flat = torch.flatten(x, 0, 1)
        l = torch.flatten(l, 0, 1)
        cat_tensor = torch.cat((x_flat, l), 1)
        input_imputation = cat_tensor.view((b, s, -1))
        x_predicted = self.imputate_model(input_imputation)[0]
        return x * m + (1 - m) * x_predicted
