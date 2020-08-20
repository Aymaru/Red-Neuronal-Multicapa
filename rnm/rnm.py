from layer import layer
from function import function

import copy
import numpy as np

hidden_neurons = 10

class RNM:
    

    def __init__(self, input_size, output_size):
        print("This is class RNM!\n")
        
        fun = function.Function()
        ##Add the layers
        self.layers = []

        self.layers.append(layer.Layer(input_size, hidden_neurons, fun.sigmoid))
        self.layers.append(layer.Layer(hidden_neurons, output_size, fun.identity))

        self.is_training = False
        

    def predict(self, x):
        forward_values = []
        partial_forward_values = []
        x_1 = []
        x_1 = np.append(x,1)
        partial_forward_values.append(x_1)

        for i in range(0,len(self.layers)):
            x = self.layers[i].forward(x)
            x_1 = []
            x_1 = np.append(x,1)
            partial_forward_values.append(x_1)

        if (self.is_training):
            forward_values = partial_forward_values
        else:
            print ("pred" ,x)
        return x, forward_values
    
    def train(self,X,Y,epochs,batch_size,lr):
        self.is_training = True

        for _ in range(0,epochs):
            training_set = X
            targets = Y

            while(len(training_set)>0):

                training_set, targets = self.grad_target(training_set,targets,batch_size,lr)
        
        self.is_training = False

        return
    
    def grad_target(self,X,Y,batch_size,lr):
        batch_size = min(len(X),batch_size)
        
        forward_values = {}
        predictions = np.ndarray(shape=(batch_size,1),dtype=(float))
        target = np.ndarray(shape=(batch_size,1),dtype=(float))

        for i in range(0,batch_size):
            pred, forward_val = self.predict(X[i])
            forward_values[i] = forward_val
            
            predictions[i] = pred
            target[i] = Y[i]

        grad = predictions - target
        
        for i in range(len(self.layers)-1,0,-1):
            w_grad = 0
            x_grad = np.array([])

            for j in range(0,len(forward_values)):

                forward_1 = forward_values[j][i]
                forward_2 = forward_values[j][i+1][:-1]
                
                x,w = self.layers[i].backward( grad[j], forward_1, forward_2)
                w_grad += w

                x_grad = np.append( x_grad, np.transpose(x) )
            
            for z in range(0,len(x_grad)):
                x_grad[z] = np.delete(x_grad[z],-1)
            
            # self.layers[i].weights -= np.transpose(w_grad * lr)
            self.layers[i].weights -= np.transpose( lr * w_grad )

            grad = x_grad

        training_set = X[batch_size::]
        targets = Y[batch_size::]

        return training_set, targets

