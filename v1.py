#Nueral Network from scratch (with a working non mechincal keyboard)

import math
import random

# Sigmoid function:    
#   This is used to:
#       convert any real-valued number into a value between 0 and 1.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
#       Finds the graident (put simply -> steepness of a cruve)
#           This shows us the sensivity:
#               Steep curve -> small input change -> big output change
#               Flat curve -> small input change -> tiny output change
def sigmoid_derivative(x):
    return x * (1 - x)  # we do not need σ since x is already sigmoid output ;)

# Initialize weights randomly
#   we do this since if the numbers are not random then we cannot break symitry
#   meaning the network would be at a standstill

#6 wieghts
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
w3 = random.uniform(-1, 1)
w4 = random.uniform(-1, 1)
w5 = random.uniform(-1, 1)
w6 = random.uniform(-1, 1)

#3 biases
b1 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)
b3 = random.uniform(-1, 1)

#this is a variable that gets ajusted by gradient but every epochs
learning_rate = 0.1

#total epochs
total_epochs = 10000

# Training data (XOR example)
training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

#1000 epochs
for epoch in range(1, total_epochs + 1):
    #basically for every tuple in the training data
    for x1, x2, y in training_data:

        # ---- Forward pass (passing thru the network) ----
        z1 = w1*x1 + w2*x2 + b1 #   wieghted input to first nueron of x1  >>>>  Finds out how activated the nueron is!
        a1 = sigmoid(z1)# squashes z1 between 1 and 0 ready for next nueron

        z2 = w3*x1 + w4*x2 + b2# weighted input to first nueron of x2
        a2 = sigmoid(z2)    # squishes between 1 and 0 ready for next one

        #output layer
        z3 = w5*a1 + w6*a2 + b3#weightde outputs of hidden layer
        y_hat = sigmoid(z3)#y_hat is the nuerol networks prediction

        # ---- Backprop ----
        error = y_hat - y  #calculates how far off the prediction was to the actual result

        d_yhat = error  #assigning to new variable for cleaness
        d_z3 = d_yhat * sigmoid_derivative(y_hat) #gradient of the loss: This tells us how much z3 (and its weights) should change to reduce error

        # Output layer gradients -> calucating relitivaly how much to change each wieght and bias in the output neuron
        dw5 = d_z3 * a1 #   tells us how much wieght 5 should change by (all relivively (since it is altered by the learning rate))
        dw6 = d_z3 * a2 #    tells us how much wieght 6 should change by (all relivively (since it is altered by the learning rate))
        db3 = d_z3 #        tells you how much to nudge bias 3 to reduce error

        # Hidden layer gradients
        d_a1 = d_z3 * w5#   tells us how much a1 was responisble for the error (we can not change a1 so we have to account later)
        d_a2 = d_z3 * w6#   tells us how much a2 was responisble for the error (we can not change a1 so we have to account later)

        d_z1 = d_a1 * sigmoid_derivative(a1)    #tells us how much chaning the input to the hidden nueron will affect the output
        d_z2 = d_a2 * sigmoid_derivative(a2)#   tells us how much chaning the input to the hidden nueron will affect the output

        dw1 = d_z1 * x1 #   tells us how much wieght 1 should change by (all relivively (since it is altered by the learning rate))
        dw2 = d_z1 * x2 #   tells us how much wieght 2 should change by (all relivively (since it is altered by the learning rate))
        db1 = d_z1#         tells you how much to nudge bias 1 to reduce error

        dw3 = d_z2 * x1 #   tells us how much wieght 3 should change by (all relivively (since it is altered by the learning rate))
        dw4 = d_z2 * x2 #   tells us how much wieght 4 should change by (all relivively (since it is altered by the learning rate))
        db2 = d_z2#         tells you how much to nudge bias 2 to reduce error

        # ---- Update weights & baises ----
        #       calculating with the learning rate how much we will change the values
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        w3 -= learning_rate * dw3
        w4 -= learning_rate * dw4
        w5 -= learning_rate * dw5
        w6 -= learning_rate * dw6

        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        b3 -= learning_rate * db3
        
    #show percentage complete of training
    if epoch % 50 == 0 or epoch == total_epochs:
        percent = (epoch / total_epochs) * 100
        print(f"Training: {percent:.1f}% complete")




# Test
for x1, x2, y in training_data: #looping over training

#Forward pass
    z1 = w1*x1 + w2*x2 + b1
    a1 = sigmoid(z1)

    z2 = w3*x1 + w4*x2 + b2
    a2 = sigmoid(z2)

    z3 = w5*a1 + w6*a2 + b3

#forward pass to output
    y_hat = sigmoid(z3)

# Convert prediction to difinite answer
    answer = "1" if y_hat >= 0.5 else "0"
    print(f"For inputs ({x1}, {x2}), the network predicts: {answer} (raw: {y_hat:.3f})")