from .abstract_network import AbstractNet
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from easydict import EasyDict


class AbstractRNN(AbstractNet):
    def __init__(self, config):
        self.config = config
        super(AbstractRNN, self).__init__(checkpoint=self.config.experiment.output_dir,
                                          gpu=self.config.experiment.gpu)

        self.input_dimensions = self.config.data.input_dimensions
        self.hidden_size = self.config.network.hidden_size
        self.gpu = self.config.experiment.gpu
        self.impute = self.config.model.impute_missing
        self.bidirectional = self.config.model.bidirectional
        self.dropout = self.config.model.dropout
        self.learn_hidden_state = self.config.model.learn_hidden_state
        self.output_cell_type = self.config.model.output_cell
        self.optim_rnn = self.config.model.enable_optimization
        self.directional_mult = 2 if self.bidirectional else 1
        self.batch_first = True
        self.hidden_dimension = self.directional_mult * self.hidden_size

        if self.impute:
            self.init_imputation_model()

    @property
    def hidden_states(self):
        params = []
        all_attrs = self._parameters
        for attr in all_attrs:
            if attr.startswith('h0') or attr.startswith('c0'):
                params.append(all_attrs[attr])
        return params

    def init_imputation_model(self):
        self.imputate_model = nn.RNN(self.input_dimensions + 1,
                                     self.input_dimensions // self.directional_mult,
                                     num_layers=2,
                                     nonlinearity=self.config.data.imputation_activation,
                                     bidirectional=self.bidirectional, batch_first=self.batch_first)

        if self.learn_hidden_state:
            self.h0_imp = nn.Parameter(
                torch.zeros(2 * self.directional_mult, 1, self.input_dimensions // self.directional_mult))

    @property
    def prediction_weights(self):
        alls = list(self.parameters())
        for hc in self.hidden_states + self.imputation_weights:
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

    def create_model(self, conf, input_size, output_size=None, name=''):
        if name:
            name = '_' + name

        config = EasyDict(conf.copy())
        models = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        if config.cell_type.lower() in models:
            model = models[config.cell_type.lower()]
        else:
            raise ValueError("Unexpected value for inner model %s (expected GRU, LSTM or RNN, got %s)" % (name,
                                                                                                          config.cell_type))

        if config.cell_type.lower() in ['gru', 'lstm']:
            if 'nonlinearity' in config:
                del config['nonlinearity']

        self.create_hidden_variable('h0_i' + name, config.num_layers * self.directional_mult, config.hidden_size)
        if config.cell_type.lower() == 'lstm':
            self.create_hidden_variable('c0_i' + name, config.num_layers * self.directional_mult, config.hidden_size)

        del config['cell_type']
        inner_model = model(input_size,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            **config)

        setattr(self, 'inner_model' + name, inner_model)

        if output_size is not None:
            if self.output_cell_type.lower() == 'fc':
                output_cell = nn.Linear(config.hidden_size * self.directional_mult, output_size)
            elif self.output_cell_type.lower() == 'rnn':
                output_cell = nn.RNN(config.hidden_size * self.directional_mult,
                                     output_size,
                                     batch_first=True,
                                     nonlinearity=self.config.model.output_activation)

                self.create_hidden_variable('h0_o' + name, 1, output_size)

            else:
                raise ValueError("Unexpected value for output cell %s (expected RNN or FC got %s)" % (name,
                                                                                                      self.output_cell_type))
            setattr(self, 'output_cell' + name, output_cell)

    def create_hidden_variable(self, name, num_layers, hidden_size):
        if self.learn_hidden_state:
            param = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
            setattr(self, name, param)


class LSTMNet(AbstractRNN):
    def __init__(self, config):
        super(LSTMNet, self).__init__(config)

        if self.config.network.num_layers == 1:
            self.dropout = 0

        self.create_model(self.config.network, self.input_dimensions, self.config.data.nb_output)

    def forward(self, x, seqs_size=None, *args, **kwargs):
        b = x.size(0)
        if self.optim_rnn:
            x = pack_padded_sequence(x, seqs_size, batch_first=self.batch_first, enforce_sorted=False)

        if self.learn_hidden_state:
            if self.config.network.cell_type == 'lstm':
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
            outs = self.output_cell(l_out).view(-1, time_step, self.config.data.nb_output)
            return outs
        elif self.output_cell_type == 'rnn':
            if self.learn_hidden_state:
                out, _ = self.output_cell(lstm_out, self.h0_o.repeat(1, b, 1))
            else:
                out, _ = self.output_cell(lstm_out)
            return self.unpad(out)


class MultiTaskRNNet(AbstractRNN):
    """
    This class has three outputs:
    - TC vs ETC: nb_channels = 2
    - TC classification: nb_channels = 4 + 1 else
    - Central Pressure Regression
    """

    def __init__(self, config):
        super(MultiTaskRNNet, self).__init__(config)

        ## Shared model
        self.create_model(self.config.network, self.input_dimensions, self.hidden_size)
        self.hidden_dimension = self.directional_mult * self.hidden_size
        self.create_model(self.config.tcXetc, self.hidden_dimension, 2, 'tcXetc')
        self.create_model(self.config.tcClass, self.hidden_dimension, 4, 'tcClass')
        self.create_model(self.config.centralPressure, self.hidden_dimension, 1, 'centralPressure')

    def forward_task(self, x, b, inner_cell_type, outer_cell_type, task_name, nb_output=None):

        if task_name:
            task_name = '_' + task_name

        inner_cell = getattr(self, 'inner_model' + task_name)

        if self.learn_hidden_state:
            h0 = getattr(self, 'h0_i' + task_name)
            if inner_cell_type == 'lstm':
                c0 = getattr(self, 'c0_i' + task_name)
                inner_out, _ = inner_cell(x, (h0.repeat(1, b, 1), c0.repeat(1, b, 1)))
            else:
                inner_out, _ = inner_cell(x, h0.repeat(1, b, 1))
        else:
            inner_out, _ = inner_cell(x)

        output_cell = getattr(self, 'output_cell' + task_name)

        if outer_cell_type.lower() == 'rnn':
            if self.learn_hidden_state:
                h0 = getattr(self, 'h0_o' + task_name)
                out, _ = output_cell(x, h0.repeat(1, b, 1))
            else:
                out, _ = output_cell(x)

            return self.unpad(out)

        else:
            out = self.unpad(inner_out)
            time_step = out.size(1)
            l_out = out.view(-1, self.hidden_dimension)
            outs = output_cell(l_out).view(-1, time_step, nb_output)
            return outs

    def forward(self, x, seqs_size):
        b = x.size(0)
        if self.optim_rnn:
            x = pack_padded_sequence(x, seqs_size, batch_first=self.batch_first, enforce_sorted=False)

        if self.learn_hidden_state:
            if self.config.network.cell_type == 'lstm':
                first_out, _ = self.inner_model(x, (self.h0_i.repeat(1, b, 1), self.c0_i.repeat(1, b, 1)))
            else:
                first_out, _ = self.inner_model(x, self.h0_i.repeat(1, b, 1))
        else:
            first_out, _ = self.inner_model(x)

        tcXetc_out = self.forward_task(first_out, b, self.config.tcXetc.cell_type.lower(),
                                       self.output_cell_type.lower(), 'tcXetc', 2)

        tcClass_out = self.forward_task(first_out, b, self.config.tcClass.cell_type.lower(),
                                       self.output_cell_type.lower(), 'tcClass', 4)

        pressure_out = self.forward_task(first_out, b, self.config.centralPressure.cell_type.lower(),
                                        self.output_cell_type.lower(), 'centralPressure', 1)

        return tcXetc_out, tcClass_out, pressure_out