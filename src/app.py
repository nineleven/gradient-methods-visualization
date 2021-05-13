import sys

from PyQt5.QtWidgets import QApplication

from mainwindow import MainWindow

from types import TracebackType

from typing import Any, Type


def except_hook(cls: Type[BaseException],
                exception: BaseException,
                traceback: TracebackType) -> Any:
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    '''
    making qt output errors to stdout
    '''
    sys.excepthook = except_hook
    
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
