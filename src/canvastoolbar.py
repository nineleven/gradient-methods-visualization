from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp

import numpy as np

from pathlib import Path

from canvas import Canvas
from utils import get_logger


logger = get_logger(Path(__file__).name)


class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas):
        logger.debug('Creating CanvasToolBar object')
        
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
        logger.debug('run button clicked')
        
        if not self.led_x0.text():
            logger.debug('lineedit is empty')
            return

        x0 = float(self.led_x0.text())

        history = x0 + np.array([[1, 1], [1, 5], [3, 4], [2, 2], [1, 1]])

        self.canvas.update_history(history, update_axes=True)
