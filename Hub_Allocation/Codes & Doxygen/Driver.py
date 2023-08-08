"""
This code is the driver code for the clustering application created to run.
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Heuristics import Heuristics

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()

    Oop_Proje = Heuristics(MainWindow)  
    

    MainWindow.show()
    sys.exit(app.exec_())
