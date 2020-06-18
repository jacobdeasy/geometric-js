"""Utils module for notebooks."""

import matplotlib as mpl


class PlotParams():
    def __init__(self, labelsize=14):
        self.labelsize = labelsize

    def set_params(self):
        mpl.rc('font', family='serif', size=15)
        # mpl.rc('text', usetex=True)
        mpl.rcParams['axes.linewidth'] = 1.3
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['xtick.minor.size'] = 5
        mpl.rcParams['ytick.minor.size'] = 5
        mpl.rcParams['xtick.labelsize'] = self.labelsize
        mpl.rcParams['ytick.labelsize'] = self.labelsize
