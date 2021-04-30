from PyQt5.QtWidgets import QWidget, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

from pathlib import Path

import numpy as np

from utils import get_logger


logger = get_logger(Path(__file__).name)


CONTOUR_ZORDER = 1
HISTORY_ZORDER = 5


class Canvas(QWidget):

    def __init__(self):
        logger.debug('Creating Canvas object')
        
        super().__init__()

        self.margin_coef = 0.05
        
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

    def plot_contour(self):
        logger.debug('Plotting contour')
        
        xs = np.linspace(*self.ax.get_xlim(), 50)
        ys = np.linspace(*self.ax.get_ylim(), 50)

        X, Y = np.meshgrid(xs, ys)

        Z = np.empty_like(X)

        for row_n in range(X.shape[0]):
            for col_n in range(X.shape[1]):
                Z[row_n][col_n] = self.function(X[row_n][col_n], Y[row_n][col_n])

        Z_pos = 1+Z-np.min(Z)

        self.ax.contour(X, Y, Z_pos, levels=np.logspace(0, 2, 30), norm=LogNorm(),
                        cmap=plt.cm.jet, alpha=0.5, zorder=CONTOUR_ZORDER)

    def update_axes(self):
        logger.debug('Updating axes')

        assert self.function
        assert self.gradient
        
        self.ax.clear()

        x_lims, y_lims = self.compute_limits()
        self.ax.set_xlim(*x_lims)
        self.ax.set_ylim(*y_lims)

        self.plot_contour()
        self.plot_quiver()

        self.canvas.draw()


    def update_history(self, history: np.array):
        logger.debug('Updating history')
        
        assert len(history) > 1
        assert len(history.shape) == 2
        assert history.shape[1] == 2
        
        self.history = history

    def update_function(self, func, grad):
        self.function = func
        self.gradient = grad
