import matplotlib.pyplot as plt
from matplotlib import animation

import matplotlib.gridspec as gridspec
from IPython.core.display import HTML
import numpy as np
import math


def animate(sequences, interval=100, blit=True, fig_size=(14,10), get_fig=False):
    if isinstance(sequences, list) or isinstance(sequences, np.ndarray):
        fig, ax = plt.subplots(1, 1)
        animate = [[ax.imshow(np.squeeze(_), cmap='gray')] for _ in sequences]

    elif isinstance(sequences, zip):
        animate = []
        for i, el in enumerate(sequences):
            seq = []
            if i == 0:
                nb_el = len(el)
                nb_col = 2
                nb_row = math.ceil(nb_el/nb_col)
                fig, ax = plt.subplots(nb_row, nb_col)

            for j in range(len(el)):
                col = int(j % 2 != 0)
                row = j//nb_col

                if nb_row==1:
                    seq.append(ax[col].imshow(np.squeeze(el[j]), cmap='gray'))
                else:
                    seq.append(ax[row, col].imshow(np.squeeze(el[j]), cmap='gray'))

            animate.append(seq)
    else:
        raise ValueError("Expected type is zip, list or numpy.ndarray, got ", type(sequences))

    fig.set_size_inches(*fig_size)

    anim = animation.ArtistAnimation(fig, animate, interval=interval, blit=blit)

    if not get_fig:
        return anim
    else:
        return anim, fig


def html_animation(sequences, interval=100, blit=True, fig_size=(14,10)):

    anim = animate(sequences, interval, blit, fig_size)
    return HTML(anim.to_html5_video())


def plot_results(batch, fig_size=(14,10)):
    if batch.shape[0] == 1 or batch.ndim in [2,3]:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.squeeze(batch), cmap='gray')

    else:
        nb_el = batch.shape[0]
        nb_col = 2
        nb_row = math.ceil(nb_el/nb_col)
        fig, ax = plt.subplots(nb_row, nb_col)

        for j in range(nb_el):
            col = int(j % 2 != 0)
            row = j//nb_col
            if nb_row==1:
                ax[col].imshow(np.squeeze(batch[j]), cmap='gray')
            else:
                ax[row, col].imshow(np.squeeze(batch[j]), cmap='gray')

    fig.set_size_inches(*fig_size)

    fig.show()


class VisuResultsClassification:
    def __init__(self, x,
                 prob,
                 groundtruth,
                 array=None,
                 std=None,
                 interval=50):
        """
        Sequence | prediction - groundtruth
        uncertainty | bar of prediction
        :param x:
        :param prob:
        :param groundtruth:
        :param nb_class:
        :param array:
        :param std:
        """
        self.x = x
        self.length = len(x)

        self.prob = prob[:self.length]
        self.pred = np.argmax(self.prob, 1)
        self.groundtruth = groundtruth[:self.length]

        self.array = self.normalize_array(np.squeeze(array))
        std = std[:self.length]
        std[0] = 0
        std[-1] = 0
        self.std = std

        self.nb_classes = self.prob.shape[1]

        self.fig = plt.figure(figsize=(14, 12))
        self.outer = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.25)

        # Images
        self.array_ax = self.fig.add_subplot(self.outer[0])
        self.image_plt = self.array_ax.imshow(self.array[0], cmap='gray')
        plt.axis('off')

        # Curves

        self.curves_ax = self.fig.add_subplot(self.outer[1])
        plt.xticks(rotation=25)

        self.curves_ax.plot(self.x, self.pred, 'r', label='Prediction')
        self.curves_ax.plot(self.x, self.groundtruth, 'g', label='Best track')
        self.curves_ax.legend()
        self.curves_ax.set_ylim(-0.1, float(self.nb_classes-1)+0.2)

        self.vertical_line = self.curves_ax.axvline(x=self.x[0], color="k", linestyle=':')

        # std

        self.std_ax = self.fig.add_subplot(self.outer[2])
        plt.xticks(rotation=25)

        self.std_ax.fill(self.x, self.std, c=(0., 0.1, 0.8, 0.3))
        self.std_ax.plot(self.x, self.std, c=(0, 0, 1), label='Uncertainty')
        self.std_ax.set_ylim(0,1)
        self.vertical_line_std = self.std_ax.axvline(x=self.x[0], color="k", linestyle=':')
        self.std_ax.legend()

        # bar
        self.bar_ax = self.fig.add_subplot(self.outer[3])

        colors = self._get_colors(self.pred[0], self.groundtruth[0])

        self.bar_ax.bar(np.arange(self.nb_classes), self.prob[0], color=colors, width=0.5)
        self.bar_ax.set_xticks(np.arange(self.nb_classes))
        self.anim = animation.FuncAnimation(self.fig, self._animate, frames=np.arange(self.length), interval=interval)

    def normalize_array(self, x):
        x -= x.min()
        x /= x.max()
        return x

    def _get_colors(self, prediction, ground_truth):
        neutral = [.7, .7, .7]
        incorrect = [.7, 0, 0]
        correct = [0, .8, 0]
        colors = [neutral]*self.nb_classes

        prediction = int(round(prediction))
        if prediction == ground_truth:
            colors[prediction] = correct
        else:
            colors[prediction] = incorrect

        return colors

    def _animate(self, i):

        self.image_plt.set_data(self.array[i])
        self.vertical_line.set_data([self.x[i], self.x[i]], [0, 1])
        self.vertical_line_std.set_data([self.x[i], self.x[i]], [0, 1])

        self.curves_ax.set_ylim(-0.1, float(self.nb_classes-1)+0.2)
        self.std_ax.set_ylim(0,1)

        plt.axis('off')
        self.bar_ax.clear()
        colors = self._get_colors(self.pred[i], self.groundtruth[i])
        self.bar_ax.bar(np.arange(self.nb_classes), self.prob[i], color=colors, width=0.5)
        self.bar_ax.set_xticks(np.arange(self.nb_classes))

    def html_anim(self):
        return HTML(self.anim.to_html5_video())


