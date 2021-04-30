from PyQt5.QtWidgets import QWidget, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

from pathlib import Path

import numpy as np

from utils import get_logger


logger = get_logger(Path(__file__).name)

NUM_X_TICKS = 50
NUM_Y_TICKS = 50

GRADIENT_ZORDER = 1
CONTOUR_ZORDER = 2
HISTORY_ZORDER = 5
INIT_APPROX_ZORDER = 6


class Canvas(QWidget):

    def __init__(self):
        logger.debug('Creating Canvas object')
        
        super().__init__()

        self.margin_coef = 0.05
        self.num_levels = None
        
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.history = np.array([])
        self.function = None
        self.gradient = None

        layout = QGridLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def compute_limits(self):
        logger.debug('Computing limits')
        
        min_x, min_y = self.history[0]
        max_x, max_y = self.history[0]

        for x, y in self.history[1:]:
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y

        w, h = max_x - min_x, max_y - min_y
        c = self.margin_coef
        
        x_lims = min_x - c * w, max_x + c * w
        y_lims = min_y - c * h, max_y + c * h

        return x_lims, y_lims

    def plot_quiver(self):
        logger.debug('Plotting quiver')
        
        # The x and y coordinates of the arrow locations
        x, y = self.history[:-1, 0], self.history[:-1, 1]
        # The x and y direction components of the arrow vectors
        u = self.history[1:, 0] - self.history[:-1, 0], 
        v = self.history[1:, 1] - self.history[:-1, 1]

        self.ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, zorder=HISTORY_ZORDER)

        self.ax.scatter(x[0], y[0], zorder=INIT_APPROX_ZORDER)

    def plot_gradient(self, X, Y):
        grad_X = np.empty_like(X)
        grad_Y = np.empty_like(X)

        for row_n in range(X.shape[0]):
            for col_n in range(X.shape[1]):
                grad_x, grad_y = self.gradient(X[row_n][col_n], Y[row_n][col_n])
                grad_norm = (grad_x**2 + grad_y**2)**0.5
                grad_X[row_n][col_n] = grad_x / grad_norm
                grad_Y[row_n][col_n] = grad_y / grad_norm

        self.ax.quiver(X, Y, grad_X, grad_Y, scale=50, width=3e-3, color='gray', alpha=0.5, zorder=GRADIENT_ZORDER)

    def plot_contour(self, X, Y):
        logger.debug('Plotting contour')

        Z = np.empty_like(X)

        max_Z = None

        for row_n in range(X.shape[0]):
            for col_n in range(X.shape[1]):
                Z[row_n][col_n] = self.function(X[row_n][col_n], Y[row_n][col_n])
                if not max_Z or Z[row_n][col_n] > max_Z:
                    max_Z = Z[row_n][col_n]

        max_Z += 1 - np.min(Z)
        Z_pos = 1 + Z - np.min(Z)

        max_z_order = np.ceil(np.log10(max_Z))

        c = self.ax.contour(X, Y, Z_pos, levels=np.logspace(0, max_z_order, self.num_levels), norm=LogNorm(),
                            cmap=plt.cm.jet, alpha=0.5, zorder=CONTOUR_ZORDER)

    def update_axes(self):
        logger.debug('Updating axes')

        if any(map(lambda x: x is None, [self.function, self.gradient, self.history])):
            logger.debug('Nothing to plot')
            return

        self.ax.clear()

        x_lims, y_lims = self.compute_limits()
        self.ax.set_xlim(*x_lims)
        self.ax.set_ylim(*y_lims)

        self.ax.set_title('BFGS')

        xs = np.linspace(*self.ax.get_xlim(), NUM_X_TICKS)
        ys = np.linspace(*self.ax.get_ylim(), NUM_Y_TICKS)
        X, Y = np.meshgrid(xs, ys)

        self.plot_gradient(X, Y)
        self.plot_contour(X, Y)
        self.plot_quiver()

        self.canvas.draw()


    def update_history(self, history):
        logger.debug('Updating history')

        history = np.array(history)
        
        assert len(history) > 1
        assert len(history.shape) == 2
        assert history.shape[1] == 2
        
        self.history = history

    def update_function(self, func, grad):
        self.function = func
        self.gradient = grad

    def update_num_levels(self, num_levels):
        assert num_levels > 0
        
        self.num_levels = num_levels
        self.update_axes()
