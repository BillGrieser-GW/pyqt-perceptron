import sys
from PyQt4 import QtCore
from PyQt4 import QtGui

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# The transfer function
hardlim = lambda n: 0 if n < 0 else 1

# =============================================================================
# Perceptron Class
# =============================================================================
class SingleNeuronPerceptron():
    """A simple implementation of a perceptron"""
    def __init__(self, R=2, S=1):
        
        self.R = R # Num input dimensions
        self.Weights = np.random.random(self.R)
        self.bias=np.random.random(S)
        
    def run_forward(self, p):
        """Given an input of dimension R, run the network"""
        return hardlim(self.Weights * p + self.bias )
    
    def train_one_iteration(self, p, t):
        """Given one input of dimension R and its target, perform one training iteration.
        Update the weights and biases using the Perceptron learning Rule."""
        t_hat = self.run(p)
        
        self.error = t_hat - t
        
        # Adjust weights and bias based on the error from this iteration
        self.Weights = self.Weights + self.error * p.T
        self.bias = self.bias + self.error
        return self.error
        
    def find_decision_boundary(self, x):
        """Returns the corresponding y value for the input x on the decision
        boundary"""
        return -(x * self.Weights[0] + self.bias) / \
            (self.Weights[1] if self.Weights[1] != 0 else .000001)

# =============================================================================
# Main GUI
# =============================================================================
class PerceptronDemo(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Perceptron Demo')
        self.frame = QtGui.QWidget()    
        self.create_plot()
        self.layout_window()
        self.create_status_bar()   
        self.setGeometry(100,100, 640,480)
        self.pos_data = []
        self.neg_data = []
        self.net = SingleNeuronPerceptron()
        
    def create_plot(self):
        
        self.plot_figure = Figure((6.0, 4.0), dpi=100)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.setParent(self.frame)
        
        # Add a plot
        self.axes = self.plot_figure.add_subplot(111)
        self.axes.set_xlim(0,10)
        self.axes.set_ylim(0,10)
        self.axes.set_xlabel("Width", fontsize=10)
        self.axes.set_ylabel("Height", fontsize=10)
        self.pos_line, = self.axes.plot([], 'mo', label="Cat")
        self.neg_line, = self.axes.plot([], 'cs', label="Bear")
        self.decision, = self.axes.plot([], 'r-', label="Decision Boundary")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.5, 
                         numpoints=1, ncol=3)
        self.axes.set_title("Single Neuron Perceptron")
        self.plot_canvas.draw()
        
        # Add event handler for a mouseclick in the plot
        self.plot_canvas.mpl_connect('button_press_event', self.on_mouseclick)
       
    def create_status_bar(self):
        self.current_status = QtGui.QLabel("Starting")
        self.statusBar().addWidget(self.current_status, 1)
        
    def layout_window(self):
        self.run_button = QtGui.QPushButton("Run")
        self.connect(self.run_button, QtCore.SIGNAL('clicked()'), self.on_run)
        self.clear_button = QtGui.QPushButton("Reset")
        self.connect(self.clear_button, QtCore.SIGNAL('clicked()'), self.on_reset)
       
        controls = QtGui.QVBoxLayout()
        for w in (self.run_button, self.clear_button):
            controls.addWidget(w)
            controls.setAlignment(w, QtCore.Qt.AlignHCenter)
            
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.plot_canvas)
        hbox.addLayout(controls)
        self.frame.setLayout(hbox)
        self.setCentralWidget(self.frame)
            
    def draw_data(self):
        self.pos_line.set_data([x[0] for x in self.pos_data], [y[1] for y in self.pos_data])
        self.neg_line.set_data([x[0] for x in self.neg_data], [y[1] for y in self.neg_data])
        self.plot_canvas.draw()
    
    def draw_decision_boundary(self):
        lim = self.axes.get_xlim()
        X = np.linspace(lim[0], lim[1], 101)
        Y = self.net.find_decision_boundary(X)
        print(X)
        print(Y)
        self.decision.set_data(X,Y)
        self.plot_canvas.draw()
        
    def on_mouseclick(self, event):
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            if event.button==1:    
                self.pos_data.append((event.xdata, event.ydata))
            else:
                self.neg_data.append((event.xdata, event.ydata))
            self.draw_data()
        
    def on_reset(self):
        self.pos_data = []
        self.neg_data = []
        self.draw_data()
        
    def on_run(self):
        
        
        
        self.draw_decision_boundary()
        
if __name__ == "__main__":
    app=None
    app = QtGui.QApplication(sys.argv)
    main_form = PerceptronDemo()
    main_form.show()
    app.exec_()
    app = None
    