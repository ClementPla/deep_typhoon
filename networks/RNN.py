from .abstract_network import AbstractNet
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AbstractRNN(AbstractNet):
    def __init__(self, config):
        self.config = config
        super(AbstractRNN, self).__init__(checkpoint=self.config.experiment.output_dir,
                                          gpu=self.config.experiment.gpu)
        self.input_dimensions = self.config.data.input_dimensions
        self.hidden_size = self.config.network.hidden_size
        self.num_layers = self.config.network.num_layers
        self.gpu = self.config.experiment.gpu
        self.impute = self.config.model.impute_missing
        self.nb_output = self.config.data.nb_output
        self.cell_type = self.config.network.cell_type
        self.bidirectional = self.config.network.bidirectional
        self.dropout = self.config.network.dropout
        self.output_activation = self.config.network.output_activation
        self.learn_hidden_state = self.config.network.learn_hidden_state
        self.output_cell = self.config.network.output_cell
        self.optim_rnn = self.config.model.enable_optimization
        self.directional_mult = 2 if self.bidirectional else 1
        self.batch_first = True

        if self.impute:
            self.init_imputation_model()

    @property
    def hidden_states(self):
        params = []
        all_attrs = vars(self)
        for attr in all_attrs:
            if attr.startswith('h0') or attr.startswith('c0'):
                params.append(all_attrs[attr])
        return params

    def init_imputation_model(self):
        self.imputate_model = nn.RNN(self.input_dimensions + 1,
                                     self.input_dimensions // self.directional_mult,
                                     num_layers=2,
                                     nonlinearity='relu',
                                     bidirectional=self.bidirectional, batch_first=self.batch_first)

        if self.learn_hidden_state:
            self.h0_imp = nn.Parameter(
                torch.zeros(2 * self.directional_mult, 1, self.input_dimensions // self.directional_mult))

    @property
    def prediction_weights(self):
        alls = list(self.inner_model.parameters()) + list(self.output_cell.parameters())
        for hc in self.hidden_states:
            try:
                alls.remove(hc)
            except (ValueError, RuntimeError) as e:
                pass
        return alls

    @property
    def imputation_weights(self):
        if self.impute:
            alls = list(self.imputate_model.parameters())
            for hc in self.hidden_states:
                try:
                    alls.remove(hc)
                except (ValueError, RuntimeError) as e:
                    pass
            return alls
        else:
            return []

    def unpad(self, x):
        if self.optim_rnn:
            return pad_packed_sequence(x, batch_first=self.batch_first)[0]
        else:
            return x

    def imputation(self, x, m, l):
        b, s = x.size()[:2]
        x_flat = torch.flatten(x, 0, 1)
        l = torch.flatten(l, 0, 1)
        cat_tensor = torch.cat((x_flat, l), 1)
        input_imputation = cat_tensor.view((b, s, -1))
        if self.learn_hidden_state:
            x_predicted = self.imputate_model(input_imputation, self.h0_imp.repeat(1, b, 1))[0]
        else:
            x_predicted = self.imputate_model(input_imputation)[0]
        return x * m + (1 - m) * x_predicted


class LSTMNet(AbstractRNN):
    def __init__(self, config):
        super(LSTMNet, self).__init__(config)

        if self.num_layers == 1:
            self.dropout = 0
        if self.cell_type == 'lstm':
            self.inner_model = nn.LSTM(self.input_dimensions, self.hidden_size,
                                       dropout=self.dropout, num_layers=self.num_layers,
                                       batch_first=self.batch_first,
                                       bidirectional=self.bidirectional)

        elif self.cell_type == 'gru':
            self.inner_model = nn.GRU(self.input_dimensions, self.hidden_size,
                                      dropout=self.dropout, num_layers=self.num_layers,
                                      batch_first=self.batch_first,
                                      bidirectional=self.bidirectional)
        elif self.cell_type == 'rnn':
            self.inner_model = nn.RNN(self.input_dimensions, self.hidden_size,
                                      num_layers=self.num_layers,
                                      batch_first=self.batch_first, dropout=self.dropout,
                                      bidirectional=self.bidirectional)

        self.hidden_dimension = self.directional_mult * self.hidden_size

        if self.output_cell_type == 'rnn':
            self.output_cell = nn.RNN(self.hidden_size * self.directional_mult, self.nb_output,
                                      num_layers=1,
                                      batch_first=self.batch_first,
                                      nonlinearity=self.output_activation)
        elif self.output_cell_type == 'fc':
            self.output_cell = nn.Linear(self.hidden_size * self.directional_mult, self.nb_output)

        else:
            raise ValueError("Unexpected value (%s)  for output cell type"%self.output_cell_type)

        if self.learn_hidden_state:
            self.h0_i = nn.Parameter(torch.zeros(self.num_layers * self.directional_mult, 1, self.hidden_size))
            if self.cell_type == 'lstm':
                self.c0_i = nn.Parameter(torch.zeros(self.num_layers * self.directional_mult, 1, self.hidden_size))
            if self.output_cell_type == 'rnn':
                self.h0_o = nn.Parameter(torch.zeros(1, 1, self.nb_output))

    def forward(self, x, seqs_size):
        b = x.size(0)
        if self.optim_rnn:
            x = pack_padded_sequence(x, seqs_size, batch_first=self.batch_first, enforce_sorted=False)

        if self.learn_hidden_state:
            if self.cell_type == 'lstm':
                lstm_out, _ = self.inner_model(x, (self.h0_i.repeat(1, b, 1), self.c0_i.repeat(1, b, 1)))
            else:
                lstm_out, _ = self.inner_model(x, self.h0_i.repeat(1, b, 1))
        else:
            lstm_out, _ = self.inner_model(x)

        if self.output_cell_type == 'direct':
            return self.unpad(lstm_out)
        elif self.output_cell_type == 'fc':
            out = self.unpad(lstm_out)
            time_step = out.size(1)
            l_out = out.view(-1, self.hidden_dimension)
            outs = self.output_cell(l_out).view(-1, time_step, self.nb_output)
            return outs
        elif self.output_cell_type == 'rnn':
            if self.learn_hidden_state:
                out, _ = self.output_cell(lstm_out, self.h0_o.repeat(1, b, 1))
            else:
                out, _ = self.output_cell(lstm_out)
            return self.unpad(out)
