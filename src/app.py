import sys
import traceback

from PyQt5.QtWidgets import QApplication

from mainwindow import MainWindow


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    sys.excepthook = except_hook
    
    app = QApplication(sys.argv)
    app.setStyle('Windows') # for memes

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
