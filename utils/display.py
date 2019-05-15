import matplotlib.pyplot as plt
from matplotlib import animation

import matplotlib.gridspec as gridspec
from IPython.core.display import HTML
import numpy as np
import math


def animate(sequences, interval=100, blit=True, fig_size=(14, 10), get_fig=False):
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
                nb_row = math.ceil(nb_el / nb_col)
                fig, ax = plt.subplots(nb_row, nb_col)

            for j in range(len(el)):
                col = int(j % 2 != 0)
                row = j // nb_col

                if nb_row == 1:
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


def html_animation(sequences, interval=100, blit=True, fig_size=(14, 10)):
    anim = animate(sequences, interval, blit, fig_size)
    return HTML(anim.to_html5_video())


def plot_results(batch, fig_size=(14, 10)):
    if batch.shape[0] == 1 or batch.ndim in [2, 3]:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.squeeze(batch), cmap='gray')

    else:
        nb_el = batch.shape[0]
        nb_col = 2
        nb_row = math.ceil(nb_el / nb_col)
        fig, ax = plt.subplots(nb_row, nb_col)

        for j in range(nb_el):
            col = int(j % 2 != 0)
            row = j // nb_col
            if nb_row == 1:
                ax[col].imshow(np.squeeze(batch[j]), cmap='gray')
            else:
                ax[row, col].imshow(np.squeeze(batch[j]), cmap='gray')

    fig.set_size_inches(*fig_size)
    fig.show()


import matplotlib
import matplotlib.cm


class VisuResultsClassification:
    def __init__(self,
                 x,
                 sequences=None,
                 bar=None,
                 graph=None,
                 fill=None,
                 interval=50,
                 figsize=(14, 12),
                 sequences_titles=None,
                 graph_titles=None,
                 fill_titles=None):
        """
        Sequence
        :param x:
        :param sequences: List of arrays to be plotted as animated sequences
        :param bar: List of tuple of arrays to be plotted as bar
        :param graph: List of arrays to be plotted as curves
        :param fill: List of arrays to be plotted as filled  curves
        For graph and fill arguments, instead of a list of lists, you can provide a list of dict, each dict being plot on
        the same graph, with the key being the label legend
        For bar argument, each tuple should be organized as: (yProba, yGroundtruth, nb_class, *labels[optional])
        """
        assert (bar is None and graph is None and fill is None,
                "You have to provide at least one of the following argument: bar, graph, fill")

        bar = self.init_arg(bar)
        graph = self.init_arg(graph)
        sequences = self.init_arg(sequences)
        sequences_titles = self.init_arg(sequences_titles)
        fill = self.init_arg(fill)

        self.x = x
        self.length = len(x)
        self.sequences = [self.normalize_array(np.squeeze(array)) for array in sequences]
        self.fig = plt.figure(figsize=figsize)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        cmap = matplotlib.cm.get_cmap('tab20')
        self.colors = [cmap(norm(_)) for _ in np.arange(0, 20, 1)]

        nb_plots = len(sequences) + len(graph) + len(bar) + len(fill)
        self.outer = gridspec.GridSpec(math.ceil(nb_plots / 2), 2, wspace=0.1, hspace=0.25)

        iter_subplot = 0
        # Images
        self.array_axs = []
        self.image_plts = []

        for seq in self.sequences:
            self.array_axs.append(self.fig.add_subplot(self.outer[iter_subplot]))
            self.image_plts.append(self.array_axs[-1].imshow(seq[0], cmap='gray'))
            iter_subplot += 1
            plt.axis('off')

        for j, title in enumerate(sequences_titles):
            self.array_axs[j].title.set_text(title)

        # Curves
        self.graph_axs = []
        self.graph_vertical_lines = []
        for arrays in graph:
            graph_ax = self.fig.add_subplot(self.outer[iter_subplot])
            if isinstance(arrays, dict):
                for key in arrays:
                    graph_ax.plot(self.x, self.pad_missing_value(arrays[key]), label=key, color=self._get_new_color())
            else:
                graph_ax.plot(self.x, self.pad_missing_value(arrays), color=self._get_new_color())

            iter_subplot += 1
            plt.xticks(rotation=25)
            graph_ax.legend()
            self.graph_vertical_lines.append(graph_ax.axvline(x=self.x[0], color='k', linestyle=':'))
            self.graph_axs.append(graph_ax)

        for j, title in enumerate(graph_titles):
            self.graph_axs[j].title.set_text(title)

        self.fill_axs = []
        self.fill_vertical_lines = []
        for arrays in fill:
            fill_ax = self.fig.add_subplot(self.outer[iter_subplot])
            if isinstance(arrays, dict):
                for key in arrays:
                    color = self._get_new_color()
                    fill_ax.plot(self.x, self.pad_missing_value(arrays[key]), label=key, color=color)
                    color[-1] = 0.5
                    filledArray = arrays[key]
                    filledArray[0] = 0
                    filledArray[-1] = 0
                    fill_ax.fill(self.x, self.pad_missing_value(filledArray), color=color)
            else:
                color = self._get_new_color()
                fill_ax.plot(self.x, self.pad_missing_value(arrays), color=color)
                color[-1] = 0.5
                arrays[0] = 0
                arrays[-1] = 0
                fill_ax.fill(self.x, self.pad_missing_value(arrays), color=color)
            plt.xticks(rotation=25)
            fill_ax.legend()
            self.fill_vertical_lines.append(fill_ax.axvline(x=self.x[0], color='k', linestyle=':'))
            self.fill_axs.append(fill_ax)
            iter_subplot += 1
        
        for j, title in enumerate(fill_titles):
            self.fill_axs[j].title.set_text(title)

        # bar
        self.bars = bar
        self.bar_axs = []
        for arrays in self.bars:
            bar_ax = self.fig.add_subplot(self.outer[iter_subplot])
            self.fill_bar(bar_ax, arrays, 0)
            self.bar_axs.append(bar_ax)
            iter_subplot += 1

        self.anim = animation.FuncAnimation(self.fig, self._animate,
                                            frames=np.arange(self.length),
                                            interval=interval)

    def init_arg(self, arg):
        if arg is None:
            arg = []
        elif not isinstance(arg, list):
            arg = [arg]
        return arg

    def pad_missing_value(self, array):
        missing_value = max(0, self.length-len(array))
        return np.pad(array, (0, missing_value), 'constant')

    def fill_bar(self, bar_ax, tuple_array, timestamp):
        labels = None
        nb_class = tuple_array[2]
        if len(tuple_array) == 4:
            labels = tuple_array[-1]
            tuple_array = tuple_array[:2]

        proba = tuple_array[0]
        gt = tuple_array[1]
        if timestamp < len(gt):
            pred = np.argmax(proba, axis=1)[timestamp]
            gt = gt[timestamp]
            color = self._get_bar_colors(pred, gt, nb_class)
            bar_ax.bar(np.arange(nb_class), proba[timestamp], color=color, width=0.5)
            bar_ax.set_xticks(np.arange(nb_class))
            if labels is not None:
                bar_ax.set_xticklabels(labels)
        else:
            color = [.7, .7, .7]
            bar_ax.bar(np.arange(nb_class), np.zeros(nb_class), color=color, width=0.5)
            bar_ax.set_xticks(np.arange(nb_class))
            if labels is not None:
                bar_ax.set_xticklabels(labels)

    def normalize_array(self, x):
        x -= x.min()
        x /= x.max()
        return x

    def _get_new_color(self):
        self.colors = np.roll(self.colors, -1)
        return self.colors[0]

    def _get_bar_colors(self, prediction, ground_truth, nb_class):
        neutral = [.7, .7, .7]
        incorrect = [.7, 0, 0]
        correct = [0, .8, 0]
        colors = [neutral] * nb_class

        prediction = int(round(prediction))
        if prediction == ground_truth:
            colors[prediction] = correct
        else:
            colors[prediction] = incorrect

        return colors

    def _animate(self, i):
        for j, plot in enumerate(self.image_plts):
            plot.set_data(self.sequences[j][i])
        for verticalLine in self.graph_vertical_lines:
            verticalLine.set_data([self.x[i], self.x[i]], [0, 1])

        for verticalLine in self.fill_vertical_lines:
            verticalLine.set_data([self.x[i], self.x[i]], [0, 1])


        plt.axis('off')
        for bar_ax, bar_array in zip(self.bar_axs, self.bars):
            bar_ax.clear()
            self.fill_bar(bar_ax, bar_array, i)

    def html_anim(self):
        return HTML(self.anim.to_html5_video())
