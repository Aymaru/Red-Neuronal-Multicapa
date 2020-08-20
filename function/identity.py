import numpy as np

class IDENTITY:

    def __init__(self):
        self.f = self.__f
        self.df = self.__df 

        # F.Sigmoid.f = @(x) 1 ./ (1 + exp(-x));
        # F.Sigmoid.df = @(x) F.Sigmoid.f(x) .* ( 1 - F.Sigmoid.f(x) );
    
    def __f(self,x):
        return x
    
    def __df(self,x):
        return 1