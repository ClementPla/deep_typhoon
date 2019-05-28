import pandas as pd
import sys
import random
import warnings
from IPython import display

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau

sys.path.insert(0, '/home/clement/code/src/deep_typhoon/')
sys.path.insert(0, '/home/clement/code/JuNNo/lib/')

from junno.j_utils import Process
from junno.j_utils.math import ConfMatrix

from utils.temporal_tools import *
from utils.datasets import *
from utils.tensors import init_cuda_sequences_batch
from networks.RNN import MultiTaskRNNet
from os import path


class RNNMultiTaskTrainer():
    def __init__(self, config, s_print=None, initialize=True):
        self.config = config
        self.s_print = s_print

        self.set_seed()
        if initialize:
            self.set_data()
            self.set_model()

    def _print(self, *a, **b):
        if self.s_print is None:
            print(*a, ** b)
        else:
            self.s_print(*a, **b)

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
            data = advance_time(data, self.config.experiment.prediction_avance)

        data['tcXetc'] = data['class'].apply(lambda x: 0 if int(x) != 6 else 1)
        classes = np.asarray(data['class']) - 2

        classes[classes == 4] = -1
        classes[classes == 5] = -1
        data['tcClass'] = classes

        self.datasets = split_dataframe(data, test_set_year=self.config.data.test_split_set,
                                        validation_ratio=self.config.data.validation_ratio)

        values, _ = np.histogram(self.datasets['train']['tcXetc'], len(np.unique(self.datasets['train']['tcXetc'])))
        self.class_weighting_tcXetc = (1 - np.asarray(values) / sum(values)).astype(np.float32)

        values, _ = np.histogram(self.datasets['train']['tcClass'], len(np.unique(self.datasets['train']['tcClass'])))
        self.class_weighting_tcClass = (1 - np.asarray(values[1:]) / sum(values[1:])).astype(np.float32)

        col = ['z_space', 'pressure', 'tcXetc', 'tcClass']
        if self.config.model.impute_missing:
            col += ['m', 'l']

        self.tsd_train = TyphoonSequencesDataset(self.datasets['train'], max_sequences_length,
                                                 columns=col, column_mask=True)
        self.tsd_valid = TyphoonSequencesDataset(self.datasets['validation'], max_sequences_length,
                                                 columns=col, column_mask=True)
        self.tsd_test = TyphoonSequencesDataset(self.datasets['test'], max_sequences_length,
                                                columns=col, column_mask=True)

    def set_model(self):
        self.model = MultiTaskRNNet(self.config)
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
        CElossTCxETC = nn.CrossEntropyLoss(torch.from_numpy(self.class_weighting_tcXetc).cuda(self.config.experiment.gpu))
        CElossTCClass = nn.CrossEntropyLoss(torch.from_numpy(self.class_weighting_tcClass).cuda(self.config.experiment.gpu))

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
                    y_pressure = input_train[1]  # pressure
                    y_tcXetc = input_train[2]  # tcXetc
                    y_Class = input_train[3]  # tcClass

                    if self.config.model.impute_missing:
                        m = input_train[4]
                        l = input_train[5]
                        x = self.model.imputation(x, m, l)
                    tcXetc_out, tcClass_out, pressure_out = self.model(x, seqs_size)

                    mask_seq = torch.flatten(mask_seq)

                    ### Central Pressure
                    y_pressure = torch.flatten(y_pressure)
                    output = torch.flatten(pressure_out)
                    masked_output = mask_seq * output
                    l_pressure = MSEloss(masked_output, y_pressure)

                    ### TC vs ETC
                    output = tcXetc_out.view(-1, tcXetc_out.size(-1))
                    y_tcXetc = torch.flatten(y_tcXetc)
                    masked_output = mask_seq.view(-1, 1) * output
                    l_tcXetc = CElossTCxETC(masked_output, y_tcXetc)

                    ### TC Classification
                    output = tcClass_out.view(-1, tcClass_out.size(-1))
                    y_Class = torch.flatten(y_Class)
                    only_tc = y_Class >= 0
                    masked_output = mask_seq.view(-1, 1) * output
                    l_tcClass = CElossTCClass(masked_output[only_tc], y_Class[only_tc])

                    l_total = l_tcXetc + l_pressure + l_tcClass
                    self.model.zero_grad()
                    # encoder
                    l_total.backward(retain_graph=True)
                    optimizer.step()
                    p_epoch.update(1)

                    if i and i % self.config.training.validation_freq == 0:
                        valid_loader = DataLoader(dataset=self.tsd_valid,
                                                  batch_size=self.config.hp.batch_size,
                                                  shuffle=True,
                                                  pin_memory=False)

                        metrics, output = self.test(valid_loader)
                        conf_tcXetc = metrics['tcXetc']
                        conf_tcXetc.labels = ['TC', 'ETC']
                        conf_tcClass = metrics['tcClass']
                        
                        if self.config.training.verbose:
                            self._print("Epoch %i, iteration %s, Training loss %f" % (e + 1, i, float(l.cpu())))
                            self._print(
                                "Validation: loss %f" % (validation_loss))


                            if e % self.config.training.html_disp == 0:

                                display.display(conf_tcXetc)
                                display.display(conf_tcClass)

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


    def test(self, dataloader, use_uncertain=False):

        def fill_task(i, length, pred_storage, gt_storage, std_storage, pred, y, std=None):
            y_sample = y[i, :length]
            pred_sample = pred[i, :length]
            gt_storage.append(y_sample.flatten())
            pred_storage.append(pred_sample.flatten())
            if std is not None:
                std_storage.append(std[i, :length].flatten())

        self.model.eval()

        with torch.no_grad():
            full_pred = dict(tcXetc=[], tcClass=[], pressure=[])
            full_gt = dict(tcXetc=[], tcClass=[], pressure=[])
            full_std = dict(tcXetc=[], tcClass=[], pressure=[])
            nb_sequences = 0
            loss_pressure = []
            for j, valid_batch in enumerate(dataloader):
                seqs_size = valid_batch[-1].cuda(self.config.experiment.gpu)
                nb_sequences += seqs_size.size(0)
                max_batch_length = torch.max(seqs_size)
                input_train = init_cuda_sequences_batch(valid_batch[:-1], max_batch_length, self.config.experiment.gpu)
                x = input_train[0]
                y_pressure = input_train[1].cpu().numpy()  # pressure
                y_tcXetc = input_train[2].cpu().numpy()  # tcXetc
                y_Class = input_train[3].cpu().numpy()  # tcClass

                if self.config.model.impute_missing:
                    m = input_train[4]
                    l = input_train[5]
                    x = self.model.imputation(x, m, l)

                if use_uncertain:
                    tcXetc_outs = []
                    tcClass_outs = []
                    pressure_outs = []

                    self.model.train()
                    n_iter = use_uncertain
                    for i in range(n_iter):
                        tcXetc_out, tcClass_out, pressure_out = model(x, seqs_size)
                        tcXetc_outs.append(tcXetc_out)
                        tcClass_outs.append(tcClass_out)
                        pressure_outs.append(pressure_out)

                    tcXetc_outs = torch.cat(tcXetc_outs).view(n_iter, *tcXetc_outs[-1].size())
                    tcXetc_pred = torch.mean(tcXetc_outs, dim=0)
                    tcXetc_std = torch.std(tcXetc_outs, dim=0).cpu().numpy()

                    tcClass_outs = torch.cat(tcClass_outs).view(n_iter, *tcClass_outs[-1].size())
                    tcClass_pred = torch.mean(tcClass_outs, dim=0)
                    tcClass_std = torch.std(tcClass_outs, dim=0).cpu().numpy()

                    pressure_outs = torch.cat(pressure_outs).view(n_iter, *pressure_outs[-1].size())
                    pressure_pred = torch.mean(pressure_outs, dim=0)
                    pressure_std = torch.std(pressure_outs, dim=0).cpu().numpy()

                else:
                    tcXetc_pred, tcClass_pred, pressure_pred = model(x, seqs_size)
                    pressure_std = None
                    tcClass_std = None
                    tcXetc_std = None



                seqs_size = seqs_size.cpu().numpy()
                for i, length in enumerate(seqs_size):
                    fill_task(i, length, full_pred['tcClass'],
                              full_gt['tcClass'],
                              full_std['tcClass'],
                              tcClass_pred,
                              y_Class,
                              tcClass_std)

                    fill_task(i, length, full_pred['tcXetc'],
                              full_gt['tcXetc'],
                              full_std['tcXetc'],
                              tcXetc_pred,
                              y_tcXetc,
                              tcXetc_std)

                    fill_task(i, length, full_pred['pressure'],
                              full_gt['pressure'],
                              full_std['pressure'],
                              pressure_pred,
                              y_pressure,
                              pressure_std)

                    loss_pressure.append(np.mean(np.abs(full_pred['pressure'])-full_gt['pressure']))

        for k in full_gt:
            full_gt[k] = np.squeeze(np.hstack(full_gt[k]))
            full_pred[k] = np.squeeze(np.hstack(full_pred[k]))
            if use_uncertain:
                full_std[k] = np.hstack(full_std[k])

        loss_pressure = np.mean(loss_pressure)

        prediction_tcXetc = np.argmax(full_pred['tcXetc'], axis=1)
        conf_tcXetc = ConfMatrix.confusion_matrix(full_gt['tcXetc'], prediction_tcXetc)

        prediction_tcClass = np.argmax(full_pred['tcClass'], axis=1)
        conf_tcClass = ConfMatrix.confusion_matrix(full_gt['tcClass'], prediction_tcClass)

        metrics = dict(pressure=loss_pressure, tcXetc=conf_tcXetc, tcClass=conf_tcClass)
        model_outputs = dict(pressure=full_pred['pressure'],
                             tcXetc=full_pred['tcXetc'],
                             tcClass=full_pred['tcClass'])
        std_outputs = dict(pressure_std=full_std['pressure'],
                           tcXetc_std=full_std['tcXetc'],
                           tcClass_std=full_std['tcClass'])
        if use_uncertain:
            return metrics, model_outputs, std_outputs
        else:
            return metrics, model_outputs

    def evaluate(self, use_uncertain=100):

        test_loader = DataLoader(dataset=self.tsd_test, batch_size=self.config.hp.batch_size,
                                 shuffle=False, pin_memory=False)

        self.model.load(self.config.experiment.output_dir, load_most_recent=True)
        if use_uncertain:
            pred, gt, loss, std = self.test(test_loader, use_uncertain)
        else:
            pred, gt, loss = self.test(test_loader, use_uncertain)

        self._print("Test: loss %f" % loss)
        if use_uncertain:
            return loss, pred, std
        else:
            return loss, pred
