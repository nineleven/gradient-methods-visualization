from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QLabel, \
     QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp, Qt

import numpy as np

from pathlib import Path

import sympy

from canvas import Canvas
from utils import get_logger


logger = get_logger(Path(__file__).name)


class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas):
        logger.debug('Creating CanvasToolBar object')
        
        super().__init__()

        self.canvas = canvas

        self.__initialize_interface()

    def __initialize_interface(self):
        self.func_widget = QWidget()
        self.lbl_func = QLabel('f(x, y):')
        self.lbl_func.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.led_func = QLineEdit()
        self.led_func.setText('x**2+y**2')
        func_layout = QHBoxLayout()
        func_layout.addWidget(self.lbl_func)
        func_layout.addWidget(self.led_func)
        self.func_widget.setLayout(func_layout)

        self.x0_widget = QWidget()
        self.lbl_x0 = QLabel('x0:')
        self.lbl_x0.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.led_x0 = QLineEdit()
        validator = QRegExpValidator(QRegExp('-?[0-9]+(\.[0-9]*)?'))
        self.led_x0.setValidator(validator)
        self.led_x0.setText('-3')
        x0_layout = QHBoxLayout()
        x0_layout.addWidget(self.lbl_x0)
        x0_layout.addWidget(self.led_x0)
        self.x0_widget.setLayout(x0_layout)

        self.btn_run = QPushButton('run')
        self.btn_run.clicked.connect(self.btn_run_clicked)
        
        layout = QVBoxLayout()

        layout.addWidget(self.func_widget)
        layout.addWidget(self.x0_widget)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)

    def build_function(self):
        logger.debug('Building function')
        
        try:
            func_sp = sympy.sympify(str(self.led_func.text()))
            
            def func(x, y):
                nonlocal func_sp
                return float(func_sp.subs('x', x).subs('y', y).evalf())

            return func_sp, func
            
        except SyntaxError:
            logger.warning('Syntax error in function definition')
            
            QMessageBox.warning(self, 'Error', 'Syntax error in function definition', QMessageBox.Ok)
            return None

    def build_gradient(self, func):
        logger.debug('Building gradient')
        
        try:
            grad_sp = sympy.Matrix([sympy.diff(func, 'x'),
                                    sympy.diff(func, 'y')])

            def grad(x, y):
                nonlocal grad_sp
                return np.array(grad_sp.subs('x', x).subs('y', y).evalf(), dtype=np.float64)

            return grad
            
        except:
            logger.warning('Unable to differentiate the function')
            
            QMessageBox.warning(self, 'Error', 'Unable to differentiate the function', QMessageBox.Ok)
            return None
        
    def btn_run_clicked(self):
        logger.debug('run button clicked')
        
        if not self.led_x0.text():
            logger.debug('lineedit is empty')
            return

        x0 = float(self.led_x0.text())

        func_sympy, func = self.build_function()
        if not func:
            return
        
        grad = self.build_gradient(func_sympy)
        if not grad:
            return
            
        history = x0 + np.array([[1, 1], [1, 5], [3, 4], [2, 2], [1, 1]])

        self.canvas.update_history(history)
        self.canvas.update_function(func, grad)

        self.canvas.update_axes()
