import sys

from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, \
     QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np

import logging


WINDOW_SIZE = (800, 600)
WINDOW_POS = (100, 100)
WINDOW_TITLE = 'Test window'


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
stream_handler.setFormatter(formatter)
    
logger.addHandler(stream_handler)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.resize(*WINDOW_SIZE)
        self.move(*WINDOW_POS)
        self.setWindowTitle(WINDOW_TITLE)

        layout = QHBoxLayout()
    
        canvas = Canvas()
        toolbar = CanvasToolBar(canvas)

        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        
        self.setLayout(layout)

class Canvas(QWidget):

    def __init__(self):
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
        # The x and y coordinates of the arrow locations
        x, y = self.history[:-1, 0], self.history[:-1, 1]
        # The x and y direction components of the arrow vectors
        u = self.history[1:, 0] - self.history[:-1, 0], 
        v = self.history[1:, 1] - self.history[:-1, 1]

        self.ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1)

    def update_axes(self):
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
        assert len(history) > 1
        assert len(history.shape) == 2
        assert history.shape[1] == 2
        
        self.history = history

        if update_axes:
            self.update_axes()

class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas):
        super().__init__()

        self.canvas = canvas

        self.btn_run = QPushButton('run', parent=self)
        self.btn_run.clicked.connect(self.btn_run_clicked)
        
        self.led_x0 = QLineEdit(parent=self)
        validator = QRegExpValidator(QRegExp('-?[0-9]+(\.[0-9]*)?'))
        self.led_x0.setValidator(validator)
        

        layout = QVBoxLayout()

        layout.addWidget(self.led_x0)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)

    def btn_run_clicked(self):
        if not self.led_x0.text():
            return

        x0 = float(self.led_x0.text())
        
        history = x0 + np.array([[1, 1], [1, 5], [3, 4], [2, 2], [1, 1]])
        self.canvas.update_history(history, update_axes=True)
            

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Windows') # for memes

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
    
