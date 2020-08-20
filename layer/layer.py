
import numpy as np 

class Layer:

    def __init__(self,input_size,output_size,function):
        
        self.neurons = input_size + 1
        self.weights = np.random.rand(self.neurons,output_size)
        self.function = function


    def forward(self,x):
        #Add the bias
        x = np.append(x,1)
        return self.function.f(x @ self.weights)

    def backward(self,grad,inputs,outputs):

        g = self.function.df(outputs)
        grad = grad * g
        
        w_grad = np.transpose(np.asmatrix(inputs)) @ grad
        x_grad = self.weights @ np.transpose(np.asmatrix(grad))

        return x_grad, w_grad