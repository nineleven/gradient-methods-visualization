from PyQt5.QtWidgets import QWidget, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pathlib import Path

import numpy as np

from utils import get_logger


logger = get_logger(Path(__file__).name)


class Canvas(QWidget):

    def __init__(self):
        logger.debug('Creating Canvas object')
        
        super().__init__()

        self.margin_coef = 0.05
        
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.history = np.array([])

        layout = QGridLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def compute_limits(self):
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

        return (min_x, max_x), (min_y, max_y)

    def plot_quiver(self):
        logger.debug('Plotting quiver')
        # The x and y coordinates of the arrow locations
        x, y = self.history[:-1, 0], self.history[:-1, 1]
        # The x and y direction components of the arrow vectors
        u = self.history[1:, 0] - self.history[:-1, 0], 
        v = self.history[1:, 1] - self.history[:-1, 1]

        self.ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1)

    def update_axes(self):
        logger.debug('Updating axes')
        self.ax.clear()

        self.plot_quiver()
        (x_min, x_max), (y_min, y_max) = self.compute_limits()

        w, h = x_max - x_min, y_max - y_min
        c = self.margin_coef
        
        x_lims = x_min - c * w, x_max + c * w
        y_lims = y_min - c * h, y_max + c * h

        self.ax.set_xlim(*x_lims)
        self.ax.set_ylim(*y_lims)

        self.canvas.draw()


    def update_history(self, history: np.array, update_axes=False):
        logger.debug('Updating history')
        
        assert len(history) > 1
        assert len(history.shape) == 2
        assert history.shape[1] == 2
        
        self.history = history

        if update_axes:
            self.update_axes()
