from PyQt5.QtWidgets import QApplication, QWidget
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(800, 600)
    w.move(100, 100)
    w.setWindowTitle('Test window')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([1, 2], [1, 2])
    
    fc = FigureCanvas(fig)
    fc.setParent(w)
    
    w.show()
    
    sys.exit(app.exec_())
    
