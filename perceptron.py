import sys
from PyQt4 import QtCore
from PyQt4 import QtGui

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# The transfer function
hardlim = lambda n: 0 if n < 0 else 1
POS = 1
NEG = 0

# =============================================================================
# Perceptron Class
# =============================================================================
class SingleNeuronPerceptron():
    """A simple implementation of a perceptron"""
    def __init__(self, R=2, S=1):
        
        self.R = R # Num input dimensions
        self.S = S # Num neurons
        self.initialize_weights()
        
    def run_forward(self, p):
        """Given an input of dimension R, run the network"""
        return hardlim(self.Weights.dot(p) + self.bias )
    
    def train_one_iteration(self, p, t):
        """Given one input of dimension R and its target, perform one training iteration.
        Update the weights and biases using the Perceptron learning Rule."""
       
        t_hat = self.run_forward(p)
        self.error = t - t_hat
        #print(p, t, t_hat, self.error)
        
        # Adjust weights and bias based on the error from this iteration
        self.Weights = self.Weights + self.error * p.T
        self.bias = self.bias + self.error
        return self.error
        
    def find_decision_boundary(self, x):
        """Returns the corresponding y value for the input x on the decision
        boundary"""
        return -(x * self.Weights[0] + self.bias) / \
            (self.Weights[1] if self.Weights[1] != 0 else .000001)
            
    def initialize_weights(self):
        self.Weights = np.random.random(self.R) * 10
        self.bias=np.random.random(self.S) * 10

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
        self.setGeometry(50,50, 680,480)
        self.data = []
        self.data = []
        self.total_epochs = 0
        self.net = SingleNeuronPerceptron()
        
    def create_plot(self):
        
        self.plot_figure = Figure((8.0, 6.0), dpi=100)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_canvas.setParent(self.frame)
        #self.plot_canvas.setGeometry(0,0,5,900)
        
        # Add a plot
        self.axes = self.plot_figure.add_subplot(111)
        self.plot_figure.subplots_adjust(bottom=0.2, left=0.1)
        self.axes.set_xlim(0,10)
        self.axes.set_ylim(0,10)
        self.axes.set_xlabel("Width", fontsize=10)
        self.axes.set_ylabel("Height", fontsize=10)
        self.pos_line, = self.axes.plot([], 'mo', label="Cat")
        self.neg_line, = self.axes.plot([], 'cs', label="Bear")
        self.decision, = self.axes.plot([], 'r-', label="Decision Boundary")
        self.axes.legend(loc='lower center', fontsize=8, framealpha=0.9, 
                         numpoints=1, ncol=3, bbox_to_anchor=(0, -.24, 1, -.280), mode='expand')
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
        self.rerun_button = QtGui.QPushButton("Re-Run")
        self.connect(self.rerun_button, QtCore.SIGNAL('clicked()'), self.on_rerun)
        self.undo_click_button = QtGui.QPushButton("Undo Last Mouse Click")
        self.connect(self.undo_click_button, QtCore.SIGNAL('clicked()'), self.on_undo_mouseclick)
        self.clear_button = QtGui.QPushButton("Reset")
        self.connect(self.clear_button, QtCore.SIGNAL('clicked()'), self.on_reset)
       
        controls = QtGui.QVBoxLayout()
        for w in (self.run_button, self.rerun_button, self.undo_click_button, self.clear_button):
            controls.addWidget(w)
            controls.setAlignment(w, QtCore.Qt.AlignHCenter)
            
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.plot_canvas)
        hbox.addLayout(controls)
        self.frame.setLayout(hbox)
        self.setCentralWidget(self.frame)
            
    def draw_data(self):
        self.pos_line.set_data([x[0] for x in self.data if x[2] == POS], [y[1] for y in self.data if y[2] == POS])
        self.neg_line.set_data([x[0] for x in self.data if x[2] == NEG], [y[1] for y in self.data if y[2] == NEG])
        self.plot_canvas.draw()
    
    def draw_decision_boundary(self):
        lim = self.axes.get_xlim()
        X = np.linspace(lim[0], lim[1], 101)
        Y = self.net.find_decision_boundary(X)
        self.decision.set_data(X,Y)
        self.plot_canvas.draw()
        
    def clear_decision_boundary(self):
        self.decision.set_data([], [])
        self.plot_canvas.draw()
        
    def on_mouseclick(self, event):
        """Add an item to the plot"""
        if event.xdata != None and event.xdata != None:
            self.data.append((event.xdata, event.ydata, POS if event.button == 1 else NEG))
            self.draw_data()
            self.current_status.setText("x={0:0.2f} y={1:0.2f}".format(event.xdata, event.ydata))
        
    def on_reset(self):
        self.data = []
        self.clear_decision_boundary()
        self.net.initialize_weights()
        self.total_epochs = 0
        self.draw_data()
        
    def on_run(self):
        
        # Do 10 epochs
        for epoch in range(10):
           
            self.total_epochs += 1
            
            training = self.data.copy()
            np.random.shuffle(training)
            for d in training:
                self.net.train_one_iteration(np.array(d[0:2]), d[2])
                
            # Calculate the error for the epoch
            self.all_t_hat = np.array([self.net.run_forward(np.array(xy[0:2])) for xy in self.data])
            total_error = abs(np.array([t[2] for t in self.data]) - self.all_t_hat).sum()
            
            if total_error == 0:
                break
            
        # print("Epoch:", self.total_epochs, "Error is:", total_error)
        self.current_status.setText("Epoch: {0} Error: {1}".format(self.total_epochs, total_error))
            
        self.draw_decision_boundary()
        
    def on_rerun(self):
        self.net.initialize_weights()
        self.total_epochs = 0
        self.on_run()
    
    def on_undo_mouseclick(self):
        if len(self.data) > 1:
            self.data.pop()
            self.draw_data()
        
        
if __name__ == "__main__":
    app=None
    app = QtGui.QApplication(sys.argv)
    main_form = PerceptronDemo()
    main_form.show()
    app.exec_()
    app = None
    