import numpy as np

class SIGMOID:

    def __init__(self):
        self.f = self.__f
        self.df = self.__df 

        # F.Sigmoid.f = @(x) 1 ./ (1 + exp(-x));
        # F.Sigmoid.df = @(x) F.Sigmoid.f(x) .* ( 1 - F.Sigmoid.f(x) );
    
    def __f(self,x):
        return 1 / (1 + np.exp(-x))
    
    def __df(self,x):
        r = self.__f(x)
        r1 = 1 - r
        return r + r1