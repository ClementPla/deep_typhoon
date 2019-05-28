import pandas as pd
import sys
import random
import warnings

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau

sys.path.insert(0, '/home/clement/code/src/deep_typhoon/')
sys.path.insert(0, '/home/clement/code/JuNNo/lib/')

from junno.j_utils import log, Process
from utils.temporal_tools import *
from utils.datasets import *
from utils.tensors import init_cuda_sequences_batch
from networks.RNN import LSTMNet
from os import path


class RNNRegressionTrainer():
    def __init__(self, config, s_print=None, initialize=True):
        self.config = config
        self.s_print = s_print

        self.set_seed()
        if initialize:
            self.set_data()
            self.set_model()

    def set_seed(self):
        seed = self.config.experiment.seed
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_data(self):
        DB_PATH = self.config.data.db_path
        DATA_PATH = path.join(DB_PATH, self.config.data.input_file)
        data = pd.read_pickle(DATA_PATH)
        data.drop(['sequences', 'indexes'], axis=1, inplace=True)
        # Add time interval (inplace)
        add_time_interval(data)
        # Get the size of the longest sequence
        max_sequences_length = get_sequence_max_length(data)
        data = data.astype({"m": np.float32, "l": np.float32})
        if self.config.experiment.prediction_avance:
            if self.config.model.bidirectional:
                warnings.warn('For predicting value, you should not use bidirectional models!')
            data = advance_time(data, self.config.experiment.prediction_avance, 'pressure',
                                self.config.experiment.predict_all_timestep)

        self.datasets = split_dataframe(data, test_set_year=self.config.data.test_split_set,
                                        validation_ratio=self.config.data.validation_ratio)

        col = ['z_space', 'pressure']
        if self.config.model.impute_missing:
            col += ['m', 'l']

        self.tsd_train = TyphoonSequencesDataset(self.datasets['train'], max_sequences_length,
                                                 columns=col, column_mask=True)
        self.tsd_valid = TyphoonSequencesDataset(self.datasets['validation'], max_sequences_length,
                                                 columns=col, column_mask=True)
        self.tsd_test = TyphoonSequencesDataset(self.datasets['test'], max_sequences_length,
                                                columns=col, column_mask=True)

    def set_model(self):
        self.model = LSTMNet(self.config)
        self.model.cuda(self.config.experiment.gpu)

    def train(self):

        groups = [dict(params=self.model.prediction_weights,
                       weight_decay=self.config.hp.weight_decay)]

        if self.model.hidden_states:
            groups.append({'params': self.model.hidden_states, 'weight_decay': 0})

        if self.config.model.impute_missing:
            groups.append({'params': self.model.imputation_weights, 'lr': self.config.hp.imputation_lr})

        optimizer = optim.Adam(groups, lr=self.config.hp.initial_lr,
                               betas=(self.config.hp.beta1, self.config.hp.beta2), eps=1e-08,
                               weight_decay=self.config.hp.weight_decay)

        lr_decayer = ReduceLROnPlateau(optimizer, factor=self.config.hp.decay_lr, verbose=self.config.training.verbose,
                                       patience=self.config.training.lr_patience_decay)

        MSEloss = nn.L1Loss()

        for e in range(self.config.hp.n_epochs):
            train_loader = DataLoader(self.tsd_train, batch_size=self.config.hp.batch_size, shuffle=True,
                                      pin_memory=False)
            batch_number = len(train_loader)

            with Process('Epoch %i' % (e + 1), total=batch_number) as p_epoch:
                for i, train_batch in enumerate(train_loader):
                    self.model.train()
                    seqs_size = train_batch[-1].cuda(self.config.experiment.gpu)
                    max_batch_length = torch.max(seqs_size)
                    input_train = init_cuda_sequences_batch(train_batch[:-1], max_batch_length,
                                                            self.config.experiment.gpu)
                    mask_seq = input_train[-1]
                    x = input_train[0]
                    y = input_train[1]
                    if self.config.model.impute_missing:
                        m = input_train[2]
                        l = input_train[3]
                        x = self.model.imputation(x, m, l)
                    output = self.model(x, seqs_size)

                    mask_seq = torch.flatten(mask_seq)
                    y = torch.flatten(y)
                    if self.config.experiment.predict_all_timestep:
                        output = output.view(-1, output.size(-1))
                        masked_output = torch.flatten(mask_seq.view(-1, 1) * output)

                    else:
                        output = torch.flatten(output)
                        masked_output = mask_seq * output
                    l = MSEloss(masked_output, y)
                    self.model.zero_grad()
                    # encoder
                    l.backward(retain_graph=True)
                    optimizer.step()
                    p_epoch.update(1)

                    if i and i % self.config.training.validation_freq == 0:
                        valid_loader = DataLoader(dataset=self.tsd_valid,
                                                  batch_size=self.config.hp.batch_size,
                                                  shuffle=True,
                                                  pin_memory=False)
                        full_pred, full_gt, validation_loss = self.test(self.model, valid_loader)

                        if self.config.training.verbose:
                            self._print("Epoch %i, iteration %s, Training loss %f" % (e + 1, i, float(l.cpu())))
                            self._print(
                                "Validation: loss %f" % (validation_loss))

                        validation_criteria = validation_loss

                        if e == 0:
                            min_criteria_validation = validation_criteria
                        else:
                            if validation_criteria < min_criteria_validation:
                                min_criteria_validation = validation_criteria
                                self.model.save_model(epoch=e, iteration=i, loss=validation_loss,
                                                      use_datetime=self.config.training.save_in_timestamp_folder)

            p_epoch.succeed()
            if e:
                lr_decayer.step(validation_loss)

    def _print(self, *a, **b):
        if self.s_print is None:
            print(*a, **b)
        else:
            self.s_print(*a, **b)

    def test(self, model, dataloader, use_uncertain=False):
        model.eval()
        MSEloss = nn.L1Loss(reduction='none')
        with torch.no_grad():
            full_pred = []
            full_gt = []
            full_std = []
            full_loss = []
            nb_sequences = 0
            for j, valid_batch in enumerate(dataloader):
                seqs_size = valid_batch[-1].cuda(self.config.experiment.gpu)
                nb_sequences += seqs_size.size(0)
                max_batch_length = torch.max(seqs_size)
                input_train = init_cuda_sequences_batch(valid_batch[:-1], max_batch_length, self.config.experiment.gpu)
                mask_seq = input_train[-1]
                x = input_train[0]
                y = input_train[1]
                print(y.size())
                if self.config.model.impute_missing:
                    m = input_train[2]
                    l = input_train[3]
                    x = model.imputation(x, m, l)
                if self.config.experiment.predict_all_timestep:
                    y = y.view(-1, self.config.experiment.prediction_avance)[:, -1]
                if use_uncertain:
                    outs = []

                    model.train()
                    n_iter = use_uncertain
                    for i in range(n_iter):
                        output = model(x, seqs_size)
                        if self.config.experiment.predict_all_timestep:
                            output = output[:,:,-1]

                        size = output.size()
                        outs.append(output)

                    outs = torch.cat(outs).view(n_iter, *size)
                    output = torch.mean(outs, dim=0)
                    std_out = torch.std(outs, dim=0).cpu().numpy()

                else:
                    output = model(x, seqs_size)
                    if self.config.experiment.predict_all_timestep:
                        output = output[:, :, -1]

                masked_output = mask_seq * torch.squeeze(output)

                l = MSEloss(masked_output, torch.squeeze(y)).cpu().numpy()

                pred = masked_output.cpu().numpy()

                y = y.cpu().numpy()
                seqs_size = seqs_size.cpu().numpy()
                for i, length in enumerate(seqs_size):
                    y_sample = y[i, :length]
                    pred_sample = pred[i, :length]
                    full_gt.append(y_sample.flatten())
                    full_pred.append(pred_sample.flatten())
                    full_loss.append(l[i, :length])
                    if use_uncertain:
                        full_std.append(std_out[i, :length].flatten())

        full_pred = np.hstack(full_pred)
        full_gt = np.hstack(full_gt)
        full_loss = np.mean(np.hstack(full_loss))
        if use_uncertain:
            full_std = np.hstack(full_std)

        if use_uncertain:
            return full_pred, full_gt, full_loss, full_std
        else:
            return full_pred, full_gt, full_loss

    def evaluate(self, use_uncertain=100):

        test_loader = DataLoader(dataset=self.tsd_test, batch_size=self.config.hp.batch_size,
                                 shuffle=False, pin_memory=False)

        self.model.load(self.config.experiment.output_dir, load_most_recent=True)
        if use_uncertain:
            pred, gt, loss, std = self.test(self.model, test_loader, use_uncertain)
        else:
            pred, gt, loss = self.test(self.model, test_loader, use_uncertain)

        self._print("Test: loss %f" % loss)
        if use_uncertain:
            return loss, pred, std
        else:
            return loss, pred
