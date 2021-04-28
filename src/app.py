import sys

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


WINDOW_SIZE = (800, 600)
WINDOW_POS = (100, 100)
WINDOW_TITLE = 'Test window'


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.resize(*WINDOW_SIZE)
        self.move(*WINDOW_POS)
        self.setWindowTitle(WINDOW_TITLE)

class Canvas(QWidget):

    def __init__(self):
        super().__init__()
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvas(self.fig)

        layout = QGridLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        
    def add_axes(self):
        self.ax.plot([1, 2], [1, 2])

class CanvasToolBar(QWidget):

    def __init__(self, canvas):
        super().__init__()

        self.canvas = canvas

        self.draw_random_point_button = QPushButton('draw', parent=self)

        layout = QVBoxLayout()

        layout.addWidget(self.draw_random_point_button)

        self.setLayout(layout)

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Windows') # for memes

    window = MainWindow()

    main_layout = QHBoxLayout()
    
    canvas = Canvas()
    toolbar = CanvasToolBar(canvas)

    main_layout.addWidget(canvas)
    main_layout.addWidget(toolbar)
    
    window.setLayout(main_layout)

    canvas.add_axes()
    
    window.show()
    
    sys.exit(app.exec_())
    
