from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp

import numpy as np

from pathlib import Path

from sympy.parsing.sympy_parser import parse_expr

from canvas import Canvas
from utils import get_logger


logger = get_logger(Path(__file__).name)


class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas):
        logger.debug('Creating CanvasToolBar object')
        
        super().__init__()

        self.canvas = canvas

        self.btn_run = QPushButton('run')
        self.btn_run.clicked.connect(self.btn_run_clicked)
        
        self.led_x0 = QLineEdit()
        validator = QRegExpValidator(QRegExp('-?[0-9]+(\.[0-9]*)?'))
        self.led_x0.setValidator(validator)

        self.led_func = QLineEdit()
        
        layout = QVBoxLayout()

        layout.addWidget(self.led_func)
        layout.addWidget(self.led_x0)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)

    def btn_run_clicked(self):
        logger.debug('run button clicked')
        
        if not self.led_x0.text():
            logger.debug('lineedit is empty')
            return

        x0 = float(self.led_x0.text())

        try:
            func = parse_expr(self.led_func.text())
        except SyntaxError:
            logger.debug('Syntax error in function definition')
            
            QMessageBox.warning(self, 'Error', 'Syntax error in function definition', QMessageBox.Ok)
            return
            

        history = x0 + np.array([[1, 1], [1, 5], [3, 4], [2, 2], [1, 1]])

        self.canvas.update_history(history, update_axes=True)
