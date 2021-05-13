from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QLabel, QSlider, \
    QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt, QLocale

from pathlib import Path

from canvas import Canvas
from utils import get_logger
from bfgs import bfgs
from toolbar_utils import build_function, build_gradient
from errors import Error, get_error_message


logger = get_logger(Path(__file__).name)


NUM_LEVELS_SLIDER_RANGE = (10, 100)

DEFAULT_FUNCTION = 'x**2+y**2-cos(2*x+y)'
DEFAULT_APPROXIMATION = (0.5, -0.5)
DEFAULT_PRECISION = 1e-3


class CanvasToolBar(QWidget):

    def __init__(self, canvas: Canvas) -> None:
        logger.debug('Creating CanvasToolBar object')
        
        super().__init__()

        self.canvas = canvas

        self.canvas.update_num_levels(NUM_LEVELS_SLIDER_RANGE[0])

        self.__initialize_interface()

    def __initialize_interface(self) -> None:
        logger.debug('Initializing interface')

        # num levels slider
        self.num_levels_widget = QWidget()

        self.lbl_levels = QLabel('contour levels:')
        self.lbl_num_levels = QLabel(str(NUM_LEVELS_SLIDER_RANGE[0]))
        self.lbl_num_levels.setMinimumWidth(20)
        
        self.sld_num_levels = QSlider(Qt.Horizontal) # type:ignore[attr-defined]
        self.sld_num_levels.sliderReleased.connect(self.sld_released) # type:ignore[attr-defined]
        self.sld_num_levels.valueChanged.connect(self.sld_value_changed) # type:ignore[attr-defined]
        self.sld_num_levels.setRange(*NUM_LEVELS_SLIDER_RANGE)

        num_levels_layout = QHBoxLayout()

        num_levels_layout.addWidget(self.lbl_levels)
        num_levels_layout.addWidget(self.sld_num_levels)
        num_levels_layout.addWidget(self.lbl_num_levels)

        self.num_levels_widget.setLayout(num_levels_layout)
        
        # target function widget
        self.func_widget = QWidget()
        
        self.lbl_func = QLabel('f(x, y):')
        self.lbl_func.setAlignment(Qt.AlignLeft | Qt.AlignVCenter) # type:ignore[attr-defined]
        
        self.led_func = QLineEdit()
        self.led_func.setText(DEFAULT_FUNCTION)
        
        func_layout = QHBoxLayout()
        func_layout.addWidget(self.lbl_func)
        func_layout.addWidget(self.led_func)
        
        self.func_widget.setLayout(func_layout)

        # initial approximation widget
        self.init_approx_widget = QWidget()
        
        self.lbl_x0 = QLabel('x0:')

        init_approx_validator = QDoubleValidator()
        init_approx_validator.setLocale(QLocale(QLocale.English))
        
        self.led_x0 = QLineEdit()
        self.led_x0.setValidator(init_approx_validator)
        self.led_x0.setText(str(DEFAULT_APPROXIMATION[0]))

        self.lbl_y0 = QLabel('y0:')

        self.led_y0 = QLineEdit()
        self.led_y0.setValidator(init_approx_validator)
        self.led_y0.setText(str(DEFAULT_APPROXIMATION[1]))
        
        init_approx_layout = QHBoxLayout()
        init_approx_layout.addWidget(self.lbl_x0)
        init_approx_layout.addWidget(self.led_x0)
        init_approx_layout.addWidget(self.lbl_y0)
        init_approx_layout.addWidget(self.led_y0)
        
        self.init_approx_widget.setLayout(init_approx_layout)

        # precision widget
        self.epsilon_widget = QWidget()
        
        self.lbl_epsilon = QLabel('epsilon:')
        
        epsilon_validator = QDoubleValidator(bottom=0.0) # type:ignore[call-overload]
        epsilon_validator.setLocale(QLocale(QLocale.English))
        
        self.led_epsilon = QLineEdit()
        self.led_epsilon.setValidator(epsilon_validator)
        self.led_epsilon.setText(str(DEFAULT_PRECISION))
        
        epsilon_layout = QHBoxLayout()
        epsilon_layout.addWidget(self.lbl_epsilon)
        epsilon_layout.addWidget(self.led_epsilon)

        self.epsilon_widget.setLayout(epsilon_layout)
        
        # run button
        self.btn_run = QPushButton('run')
        self.btn_run.clicked.connect(self.btn_run_clicked) # type:ignore[attr-defined]
        
        layout = QVBoxLayout()

        layout.addWidget(self.num_levels_widget, alignment=Qt.AlignTop) # type:ignore[attr-defined]
        layout.addWidget(self.func_widget, alignment=Qt.AlignTop) # type:ignore[attr-defined]
        layout.addWidget(self.init_approx_widget, alignment=Qt.AlignTop) # type:ignore[attr-defined]
        layout.addWidget(self.epsilon_widget, alignment=Qt.AlignTop) # type:ignore[attr-defined]
        layout.addWidget(self.btn_run, alignment=Qt.AlignTop) # type:ignore[attr-defined]

        self.setLayout(layout)

    def sld_value_changed(self, value: int) -> None:
        self.lbl_num_levels.setText(str(value))

    def sld_released(self) -> None:
        value = self.sld_num_levels.value()
        self.canvas.update_num_levels(value)
        
    def btn_run_clicked(self) -> None:
        logger.debug('run button clicked')
        
        if not self.led_x0.text():
            QMessageBox.warning(
                self,
                'Error',
                'Missing initial approximation',
                QMessageBox.Ok
            )
            logger.debug('missing initial approximation')
            return

        if not self.led_epsilon.text():
            QMessageBox.warning(
                self,
                'Error',
                'Missing precision',
                QMessageBox.Ok
            )
            logger.debug('missing precision')
            return

        x0 = [float(self.led_x0.text()), float(self.led_y0.text())]

        epsilon = float(self.led_epsilon.text())

        err, func_sympy, func = build_function(str(self.led_func.text()))
        if err != Error.OK:
            QMessageBox.warning(
                self,
                'Error',
                get_error_message(err),
                QMessageBox.Ok
            )
            return
        
        err, grad = build_gradient(func_sympy)
        if not grad:
            QMessageBox.warning(
                self,
                'Error',
                get_error_message(err),
                QMessageBox.Ok
            )
            return

        _, history = bfgs(grad, x0, epsilon, return_history=True)

        self.canvas.update_history(history)
        self.canvas.update_function(func, grad)

        self.canvas.update_axes()
