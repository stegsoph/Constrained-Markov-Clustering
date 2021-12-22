from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import numpy as np


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'figure.figsize': (4, 3)})
    plt.rcParams.update({'figure.dpi' : 80 })
    plt.rcParams.update({'lines.linewidth': 1.5})
    plt.rcParams.update({'lines.markersize': 8})
    plt.rcParams.update({'lines.markeredgewidth': 2})
    plt.rcParams.update({'axes.labelpad': 10})
    plt.rcParams.update({'xtick.major.width': 1.5})
    plt.rcParams.update({'xtick.major.size': 10})
    plt.rcParams.update({'xtick.minor.size': 5})
    plt.rcParams.update({'ytick.major.width': 1.5})
    plt.rcParams.update({'ytick.major.size': 10})
    plt.rcParams.update({'ytick.minor.size': 5})


    # for font settings see also https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.family': 'sans-serif'})

    # this checks if the necessary executables for rendering latex are included in your path; see also
    # https://matplotlib.org/stable/tutorials/text/usetex.html
    if find_executable('latex') and find_executable('dvipng') and find_executable('ghostscript'):
        plt.rcParams.update({'text.usetex': True})
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,amsthm}' + \
                                              r'\usepackage{siunitx}' + \
                                              r'\sisetup{detect-all}' + \
                                              r'\usepackage{helvet}' + \
                                              r'\usepackage{sansmath}' + \
                                              r'\sansmath'
    

def show_clustering(X, labels, title=None, figsize=(4, 3), subplot=None):

    init_plot_style()

    if subplot is None:
        plt.figure(figsize=figsize)

    if title is not None:
        plt.title(title)
    for idx in range(len(labels)):
        plt.plot(X[labels[:] == idx, 0], X[labels[:] == idx, 1], 'o')
        plt.tight_layout()

def show_transition_prob(P, title=None, figsize=(4, 3)):
    
    init_plot_style()

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.imshow(P, cmap=plt.get_cmap("terrain"), origin='upper')
    fig.colorbar(cs)
    if title is not None:
        ax.set_title(title)