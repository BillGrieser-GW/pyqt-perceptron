# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:45:40 2018

@author: billg_000
"""

import sys
from PyQt4 import QtGui

def window():
   
   w = QtGui.QWidget()
   b = QtGui.QLabel(w)
   b.setText("Hello World!")
   w.setGeometry(100,100,200,50)
   b.move(50,20)
   w.setWindowTitle("PyQt")
   return w
	
if __name__ == '__main__':
   app = QtGui.QApplication(sys.argv)
   w = window()
   w.minimized=False
   w.show()
   app.exec_()
   app=None