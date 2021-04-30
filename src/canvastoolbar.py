from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QLabel, QSlider, \
     QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp, Qt

from pathlib import Path

import sympy

from canvas import Canvas
from utils import get_logger


logger = get_logger(Path(__file__).name)


NUM_LEVELS_SLIDER_RANGE = (10, 100)


class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas):
        logger.debug('Creating CanvasToolBar object')
        
        super().__init__()

        self.canvas = canvas

        self.canvas.update_num_levels(NUM_LEVELS_SLIDER_RANGE[0])

        self.__initialize_interface()

    def __initialize_interface(self):
        logger.debug('Initializing interface')

        # num levels slider
        self.num_levels_widget = QWidget()

        self.lbl_levels = QLabel('contour levels:')
        self.lbl_num_levels = QLabel(str(NUM_LEVELS_SLIDER_RANGE[0]))
        self.lbl_num_levels.setMinimumWidth(20)
        
        self.sld_num_levels = QSlider(Qt.Horizontal)
        self.sld_num_levels.sliderReleased.connect(self.sld_released)
        self.sld_num_levels.valueChanged.connect(self.sld_value_changed)
        self.sld_num_levels.setRange(*NUM_LEVELS_SLIDER_RANGE)

        num_levels_layout = QHBoxLayout()

        num_levels_layout.addWidget(self.lbl_levels)
        num_levels_layout.addWidget(self.sld_num_levels)
        num_levels_layout.addWidget(self.lbl_num_levels)

        self.num_levels_widget.setLayout(num_levels_layout)
        
        # target function widget
        self.func_widget = QWidget()
        
        self.lbl_func = QLabel('f(x, y):')
        self.lbl_func.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.led_func = QLineEdit()
        self.led_func.setText('x**2+y**2')
        
        func_layout = QHBoxLayout()
        func_layout.addWidget(self.lbl_func)
        func_layout.addWidget(self.led_func)
        
        self.func_widget.setLayout(func_layout)

        self.init_approx_widget = QWidget()

        # initial approximation widget
        self.lbl_x0 = QLabel('x0:')

        init_approx_validator = QRegExpValidator(QRegExp('-?[0-9]+(\.[0-9]*)?'))
        
        self.led_x0 = QLineEdit()
        self.led_x0.setValidator(init_approx_validator)
        self.led_x0.setText('-3')

        self.lbl_y0 = QLabel('y0:')

        self.led_y0 = QLineEdit()
        self.led_y0.setValidator(init_approx_validator)
        self.led_y0.setText('-2')
        
        init_approx_layout = QHBoxLayout()
        init_approx_layout.addWidget(self.lbl_x0)
        init_approx_layout.addWidget(self.led_x0)
        init_approx_layout.addWidget(self.lbl_y0)
        init_approx_layout.addWidget(self.led_y0)
        
        self.init_approx_widget.setLayout(init_approx_layout)

        # run button
        self.btn_run = QPushButton('run')
        self.btn_run.clicked.connect(self.btn_run_clicked)
        
        layout = QVBoxLayout()

        layout.addWidget(self.num_levels_widget, alignment=Qt.AlignTop)
        layout.addWidget(self.func_widget, alignment=Qt.AlignTop)
        layout.addWidget(self.init_approx_widget, alignment=Qt.AlignTop)
        layout.addWidget(self.btn_run, alignment=Qt.AlignTop)

        self.setLayout(layout)

    def sld_value_changed(self, value):
        self.lbl_num_levels.setText(str(value))

    def sld_released(self):
        value = self.sld_num_levels.value()
        self.canvas.update_num_levels(value)

    def build_function(self):
        logger.debug('Building function')
        
        try:
            func_sp = sympy.sympify(str(self.led_func.text()))
            
            def func(x, y):
                nonlocal func_sp
                func_eval = func_sp.subs('x', x).subs('y', y).evalf()
                return float(func_eval)

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
                grad_eval = grad_sp.subs('x', x).subs('y', y).evalf()
                return list(map(float, grad_eval))

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

        x0 = [float(self.led_x0.text()), float(self.led_y0.text())]

        res = self.build_function()
        if not res:
            return
        func_sympy, func = res
        
        grad = self.build_gradient(func_sympy)
        if not grad:
            return

        x, y = x0
        history = [[0, 0], [1, 5], [3, 4], [2, 2], [1, 1]]
        history = [[x+u, y+v] for u, v in history]

        self.canvas.update_history(history)
        self.canvas.update_function(func, grad)

        self.canvas.update_axes()
