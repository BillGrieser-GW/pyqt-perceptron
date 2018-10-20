# -*- coding: utf-8 -*-

# Bill Grieser
# Machine Learning II, Monday Section
#

import numpy as np
import matplotlib.pyplot as plt

# Constants
EPOCHS = 10



# Initialize input matrix from problem statement
P_training = np.matrix([[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]])

# Initialize the target vector from problem statement
t_training = np.array([0,0,0,0,1,1,1,1])

# There are two ouput values -- 0 for Rabbit, 1 for Bear. One neuron
# is needed
S = 1
R = P_training.shape[1]

# Initialize weights to random values in a Rx1 matrix
Weights = np.random.random((R,1))

# Initialize bias vector with length that matches the number of neurons
bias = np.random.random(S)

#
# Create functions to evaluate the network
#
#
# hardlim() transfer function 
#
def hardlim(n):
    return 0 if n < 0 else 1

def run_network(input_vector):
    return hardlim(input_vector*Weights + bias )

# Initialize the errors to the length of the training data
error_training = np.zeros(P_training.shape[0])

# Initialize the network output for each iteration of an epoch
a_training = np.zeros(P_training.shape[0])

for epoch in range(EPOCHS):
    
    for obs in range(P_training.shape[0]):
        
        # Run the network
        a_training[obs] = run_network(P_training[obs])
        
        # Find the error vs. the target
        error_training[obs] = t_training[obs] - a_training[obs]
        
        # Adjust weights and bias based on the error from this iteration
        Weights = Weights + error_training[obs] * P_training[obs].T
        bias = bias + error_training[obs]

cell_text = []    
# Now that the model is trained, check each input value
for obs in range(P_training.shape[0]):
    
    a_validate = run_network(P_training[obs])
    
    cell_text.append([obs+1, P_training[obs,0],  P_training[obs,1], a_validate, t_training[obs]])
    
    #print ("Observation number:", obs, "p=", P_training[obs], "Network says:", a_validate, "Actual:", t_training[obs] )
    
# Show table
columns = ('Observation', 'p1', 'p2','Predicted Target', 'Actual Target')    
plt.figure(figsize=(7, 0.25), dpi=120)
plt.table(cellText=cell_text,colLabels=columns)
plt.axis('off')
plt.title("Network predicted target vs. actual")
plt.show()


    
#%%
import pandas as pd

ppp = pd.DataFrame(P_training, columns=['p1', 'p2'])
ppp['Target'] = t_training


#%%
# Define a range of xs to make an interesting plot
X = np.arange(-1, 5, 0.1)
Y = []
# Make the y's the form the decision boundaary for the
# x's using the equation [x,y]*Weights + bias = 0
for x in X:
    y = -(x * Weights[0,0] + bias) / Weights[1,0] 
    Y.append(y)

plt.subplots(figsize=(10, 10))
plt.ylim(-0.5, 5.5)
plt.xlim(-0.5, 5.5)
plt.axhline(y=0, color="k", alpha=0.3)
plt.axvline(x=0, color="k", alpha=0.3)
plt.plot(X, Y, "r", label="Decision Boundary")
plt.scatter(x=ppp[ppp.Target==0]['p1'], y=ppp[ppp.Target==0]['p2'], color='green',marker='o', label='Rabbit', s=90)
plt.scatter(x=ppp[ppp.Target==1]['p1'], y=ppp[ppp.Target==1]['p2'], color='blue',marker='+', label='Bear', s=90)
plt.xlabel('p1')
plt.ylabel('p2')
plt.title(r"Input values with decision boundaary", fontdict={'fontsize': 20})
plt.legend(loc='lower right')
plt.show()
