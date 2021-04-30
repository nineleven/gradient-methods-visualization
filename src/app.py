import sys

from PyQt5.QtWidgets import QApplication

from mainwindow import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Windows') # for memes

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
    
