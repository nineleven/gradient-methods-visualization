from PyQt5.QtWidgets import QWidget, QHBoxLayout

from pathlib import Path

from canvas import Canvas
from canvastoolbar import CanvasToolBar
from utils import get_logger


logger = get_logger(Path(__file__).name)


WINDOW_SIZE = (800, 600)
WINDOW_POS = (100, 100)
WINDOW_TITLE = 'Test window'


class MainWindow(QWidget):

    def __init__(self):
        logger.debug('Creating MainWindow object')
        
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
