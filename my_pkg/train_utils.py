import torch
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import math
import copy
from my_pkg.data_types import *


__all__ = [
    "TrainHistory",
    "train_model",
]


class TrainHistory:

    def __init__(self, phases: List[str], metrics: List[str]):

        self.phases = phases.copy()
        self.metrics = metrics.copy()
        self.history = {}
        for phase in phases:
            self.history[phase] = {metric: [] for metric in metrics}

    def add(self, phase, metric, value):
        self.history[phase][metric].append(value)

    def get_epochs(self, phase=None, metric=None):
        if phase is None:
            phase = self.phases[0]
        if metric is None:
            metric = self.metrics[0]
        return len(self.history[phase][metric])

    def plot(self, phases=None, metrics=None):

        if phases is None:
            phases = self.phases

        if metrics is None:
            metrics = self.metrics

        rows = len(metrics)

        fig, ax = plt.subplots(rows, 1)
        if rows == 1:
            ax = [ax]
        for i, metric in enumerate(metrics):
            y_min = [0]
            y_max = []
            for phase in phases:
                y_data = self.history[phase][metric]

                y_mean = np.nanmean(y_data)
                y_std = np.nanstd(y_data)
                y_min.append(y_mean-3*y_std)
                y_max.append(y_mean+3*y_std)
                
                epoch_num = np.arange(1, self.get_epochs(phase, metric) + 1)
                ax[i].plot(epoch_num, self.history[phase][metric], label=phase)
            ax[i].set_ylabel(metric)
            ax[i].set_ylim([np.min(y_min), np.max(y_max)])
        ax[-1].set_xlabel('epoch #')
        ax[0].legend()
        return fig, ax

    def save(self, filename):
        pickle.dump({'phases': self.phases, 'metrics': self.metrics, 'history': self.history}, open(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        s = pickle.load(open(filename, 'rb'))
        obj = cls(s['phases'], s['metrics'])
        obj.history = s['history']
        return obj


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x):

        if np.isnan(x): return

        self.n += 1
        delta = x - self.mean_
        self.mean_ += delta / self.n
        delta2 = x - self.mean_
        self.M2 += delta * delta2

    def mean(self):
        return self.mean_

    def variance(self):
        if self.n < 2:
            return 0.0
        else:
            return self.M2 / (self.n - 1)

    def stddev(self):
        return math.sqrt(self.variance())


class OnlineMetricPlot:

    def __init__(self, y_label='loss', plot_every=1, loss_names=['train']):

        COLORS = [(0, 0, 1), # blue
                  (0.96, 0.69, 0.13), # mustard
                  (1, 0, 1), # magenta
                  (0, 1, 0), # green
                ]

        self.loss_names = loss_names
        self.norm_name = loss_names[0] # used to set the range of y axis during online plot

        self.rstats = {name: RunningStats() for name in loss_names}

        self.fig, self.ax = plt.subplots(1, 1)

        self.pl = {loss_name: self.ax.plot([], [], label=loss_name, color=COLORS[i], linewidth=2)[-1] for i, loss_name in enumerate(loss_names)}
        self.loss_data = {loss_name: [] for i, loss_name in enumerate(loss_names)}

        # self.ax.legend(handles=list(self.pl.values()))
        plt.figlegend(handles=list(self.pl.values()), fontsize=16, loc='upper center', ncol=len(loss_names))
        self.ax.set_xlabel('epoch #', fontsize=13)
        self.ax.set_ylabel(y_label, fontsize=13)
        self.fixed_y_lim = False

        self.epoch_data = []
        self.plot_every = plot_every
        self.plot_count = 0

    def set_ylim(self, y_min: float, y_max: float):
        self.ax.set_ylim([y_min, y_max])
        self.fixed_y_lim = True

    def add(self, epoch: int, loss_values: Dict[str, float]):
        
        self.epoch_data.append(epoch)

        for key, value in loss_values.items():
            self.loss_data[key].append(value)
            self.rstats[key].update(value)

        self.plot_count += 1
        if self.plot_count == self.plot_every:
            self.plot_count = 0
            for key, pl in self.pl.items():
                pl.set_data(self.epoch_data, self.loss_data[key])
            self.ax.set_xlim([0, epoch + 1])
            if not self.fixed_y_lim:
                y_min = np.min([0] + [v.mean() - 1.5*v.stddev() for v in self.rstats.values()])
                y_max = np.max([v.mean() + 1.5*v.stddev() for v in self.rstats.values()])
                self.ax.set_ylim([y_min, y_max])
            plt.pause(0.002)


class TrainGuiCtrl:

    NONE = 0
    STOP_TRAIN = 1
    SAVE_MODEL = 2

    def __init__(self, win_name: str) -> None:
        self.win_name = win_name
        self._input = TrainGuiCtrl.NONE

        cv2.namedWindow(self.win_name)
        info_img = np.ones((120, 620, 3), dtype='uint8')*255
        info_img = cv2.putText(info_img, '- Press "q" to stop training', (20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2, color=(255, 0, 0))
        info_img = cv2.putText(info_img, '- Press "s" to save the best model so far', (20, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2, color=(255, 0, 0))
        cv2.imshow(self.win_name, info_img)
        cv2.waitKey(30)

    def read_input(self):
        ch = cv2.waitKey(30)
        if ch == ord('q'):
            return TrainGuiCtrl.STOP_TRAIN
        elif ch == ord('s'):
            return TrainGuiCtrl.SAVE_MODEL
        else:
            return TrainGuiCtrl.NONE
        

def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fun,
                epochs: int,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader = None,
                test_loader: torch.utils.data.DataLoader = None,
                lr_scheduler=None,
                metrics=['loss'],
                return_choice='best_val',
                grad_clip_norm=None):

    train_phases = ['train']
    data_loaders = {'train': train_loader}

    if val_loader is not None:
        train_phases.append('val')
        data_loaders['val'] = val_loader

    if test_loader is not None:
        train_phases.append('test')
        data_loaders['test'] = test_loader

    history = TrainHistory(phases=train_phases, metrics=metrics)

    return_choice = return_choice.lower()
    if return_choice not in ('best_train', 'best_val', 'last_epoch'):
        raise ValueError(f"return_choice '{return_choice}' is not in ('best_train', 'best_val', 'last_epoch')")

    if not val_loader and return_choice == 'best_val':
        return_choice = 'best_train'

    log_loss = 'loss' in metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    min_loss = math.inf
    best_weights = copy.deepcopy(model.state_dict())

    gui = TrainGuiCtrl("Model training")

    loss_plot = OnlineMetricPlot(y_label='loss', loss_names=train_phases)

    for epoch in range(epochs):

        print(f'epoch {epoch + 1}/{epochs}')

        epoch_losses = {}

        for phase in train_phases:

            is_train = phase == 'train'

            data_loader = data_loaders[phase]

            model.train(is_train)  # model.eval()

            epoch_loss = 0.

            batch_size = data_loader.batch_size
            n_data = len(data_loader.dataset)

            for inputs, outputs in data_loader:
                inputs, outputs = map(lambda x: {k: v.to(device=device, dtype=torch.float32) for k, v in x.items()}, (inputs, outputs))

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    # pred_outputs = model(inputs[InputImage.data_name])
                    pred_outputs = model(inputs)

                    loss = loss_fun(pred_outputs, outputs)

                    if is_train:
                        loss.backward()
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss.item() * batch_size / n_data

            if return_choice == 'best_train' and is_train or return_choice == 'best_val' and not is_train:
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    best_weights = copy.deepcopy(model.state_dict())

            if is_train and lr_scheduler:
                lr_scheduler.step()

            if log_loss:
                history.add(phase, 'loss', epoch_loss)

            print(f'<{phase:5s}> Loss: {epoch_loss:.5f}')
            print('-' * 30)

            epoch_losses[phase] = epoch_loss

        loss_plot.add(epoch, epoch_losses)

        print('=' * 30)

        gui_in = gui.read_input()
        if gui_in == TrainGuiCtrl.STOP_TRAIN:
            print('\33[1m\33[33mExternal interrupt. Terminating training...\33[0m')
            break
        elif gui_in == TrainGuiCtrl.SAVE_MODEL:
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(best_weights)
            best_model.train_history = history   
            best_model.save(f'best_model_{epoch+1}.bin')     
            print('\33[1m\33[34mSaved best model!\33[0m')

    if return_choice != 'last_epoch':
        model.load_state_dict(best_weights)

    model.train_history = history

    return model


# ===============================================
# ===============================================

from my_pkg.train_metrics import jaccard_index

def train_segmentation_model(model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             loss_fun,
                             epochs: int,
                             train_loader: torch.utils.data.DataLoader,
                             val_loader: torch.utils.data.DataLoader = None,
                             test_loader: torch.utils.data.DataLoader = None,
                             lr_scheduler=None,
                             metrics=['loss', 'accuracy', 'IoU'],
                             return_choice='best_val'):

    
    train_phases = ['train']
    data_loaders = {'train': train_loader}

    if val_loader is not None:
        train_phases.append('val')
        data_loaders['val'] = val_loader

    if test_loader is not None:
        train_phases.append('test')
        data_loaders['test'] = test_loader

    history = TrainHistory(phases=train_phases, metrics=metrics)

    return_choice = return_choice.lower()
    if return_choice not in ('best_train', 'best_val', 'last_epoch'):
        raise ValueError(f"return_choice '{return_choice}' is not in ('best_train', 'best_val', 'last_epoch')")

    if not val_loader and return_choice == 'best_val':
        return_choice = 'best_train'

    log_loss = 'loss' in metrics
    log_acc = 'accuracy' in metrics
    log_iou = 'IoU' in metrics

    n_classes = model.n_classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    min_loss = math.inf
    best_weights = copy.deepcopy(model.state_dict())

    gui = TrainGuiCtrl('Segmentation training')

    loss_plot = OnlineMetricPlot(y_label='loss', loss_names=train_phases)
    iou_plot = OnlineMetricPlot(y_label='mean IoU', loss_names=train_phases)
    iou_plot.set_ylim(0, 1)

    for epoch in range(epochs):

        print(f'epoch {epoch + 1}/{epochs}')

        epoch_losses = {}
        epoch_IoUs = {}

        for phase in train_phases:

            train = phase == 'train'

            data_loader = data_loaders[phase]

            model.train(train)  # model.eval()

            epoch_loss = 0.
            epoch_accuracy = 0.
            epoch_IoU = torch.zeros(n_classes, device=device)

            batch_size = data_loader.batch_size
            n_data = len(data_loader.dataset)

            for inputs, outputs in data_loader:
                rgb_img, seg_mask = map(lambda x: x.to(device), (inputs[InputImage.data_name],
                                                                 outputs[SegmentationMask.data_name]))

                # n_cols = 4
                # n_rows = int(len(rgb_img)/n_cols)
                # rgb_img = np.vstack([np.hstack([rgb_img[i].detach().cpu().numpy().transpose(1, 2, 0) for i in range(k*n_cols, (k+1)*n_cols)]) for k in range(n_rows)])
                # fig, ax = plt.subplots()
                # ax.axis('off')
                # ax.imshow(rgb_img)
                # plt.pause(0.001)
                # input('...')
                # exit(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(train):
                    pred_seg_mask = model(rgb_img)['seg_mask']
                    pred = torch.argmax(pred_seg_mask, dim=1)

                    loss = loss_fun(pred_seg_mask, seg_mask)

                    if train:
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss.item() * batch_size / n_data

                    n_pixels = seg_mask.shape[1] * seg_mask.shape[2]
                    epoch_accuracy += torch.sum(pred == seg_mask).item() / n_pixels / n_data

                    y_true = torch.nn.functional.one_hot(seg_mask, n_classes)
                    y_pred = torch.nn.functional.one_hot(pred, n_classes)
                    epoch_IoU += jaccard_index(y_true, y_pred, reduction='none') * y_true.shape[0]

            if return_choice == 'best_train' and train or return_choice == 'best_val' and not train:
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    best_weights = copy.deepcopy(model.state_dict())

            if train and lr_scheduler:
                lr_scheduler.step()

            epoch_IoU = torch.mean(epoch_IoU / n_data).item()

            if log_loss:
                history.add(phase, 'loss', epoch_loss)
            if log_acc:
                history.add(phase, 'accuracy', epoch_accuracy)
            if log_iou:
                history.add(phase, 'IoU', epoch_IoU)

            print(f'<{phase:5s}> Loss: {epoch_loss:.5f}, Accuracy: {epoch_accuracy * 100:.2f} %, IoU: {epoch_IoU * 100:.2f} %')
            print('-' * 36)

            epoch_losses[phase] = epoch_loss
            epoch_IoUs[phase] = epoch_IoU

        loss_plot.add(epoch, epoch_losses)
        iou_plot.add(epoch, epoch_IoUs)

        print('=' * 36)

        gui_in = gui.read_input()
        if gui_in == TrainGuiCtrl.STOP_TRAIN:
            print('\33[1m\33[33mExternal interrupt. Terminating training...\33[0m')
            break
        elif gui_in == TrainGuiCtrl.SAVE_MODEL:
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(best_weights)
            best_model.train_history = history   
            best_model.save(f'best_seg_model_{epoch+1}.bin')     
            print('\33[1m\33[34mSaved best model!\33[0m')

    if return_choice != 'last_epoch':
        model.load_state_dict(best_weights)

    model.train_history = history

    return model