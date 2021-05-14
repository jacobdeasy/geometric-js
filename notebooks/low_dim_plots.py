import matplotlib.pyplot as plt
import numpy as np

from utils import PlotParams
plotter = PlotParams()
plotter.set_params()


# tGJS dimensions
x = np.arange(1, 11)
y = [0.730, 0.790, 0.688, 0.609, 0.568, 0.547, 0.536, 0.518, 0.511, 0.507]
plt.plot(x, y, 'bo-')
plt.xlabel('Data dimension')
plt.ylabel(r'Final $\alpha$')
plt.show()


# tGJS components
x = np.arange(1, 11)
y = [0.806, 0.807, ]
plt.plot(x, y, 'oo-')
plt.xlabel('Data dimension')
plt.ylabel(r'Final $\alpha$')
plt.show()
