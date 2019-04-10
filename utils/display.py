import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.core.display import HTML
import numpy as np
import math


def animate(sequences, interval=100, blit=True, fig_size=(14,10), get_fig=False):
    if isinstance(sequences, list) or isinstance(sequences, np.ndarray):
        fig, ax = plt.subplots(1, 1)
        animate = [[ax.imshow(np.squeeze(_), cmap='gray') for _ in sequences]]

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
                col = int(j %2 == 0)
                row = j//nb_col
                print(col, row)
                seq.append(ax[col, row].imshow(np.squeeze(el[i]), cmap='gray'))

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




