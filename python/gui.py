# https://python.plainenglish.io/build-modern-gui-in-python-using-pyqt5-framework-d3398beeb555
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
def main():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(400,400,400,300)
    win.setWindowTitle("Pyqt5 Tutorial")
    #Button
    button = QtWidgets.QPushButton(win)
    button.setText("Hi! Click Me")
    button.move(100,100)
    # Alert
    msg = QtWidgets.QMessageBox(win)
    msg.setWindowTitle("Pyqt5 Alert")
    msg.setText("Message box in Pyqt5!")
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.exec_()
    win.show()
    sys.exit(app.exec_())

main()
#%%
