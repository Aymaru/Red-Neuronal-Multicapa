import numpy as np

class DATA:
    
    def generate_data_set_f_xy(self,ds_size,range_a,range_b):
        X = np.random.uniform(range_a,range_b,size=(ds_size,2))
        Y = np.ndarray(shape=(ds_size,1),dtype=(float))
        for i in range(0,len(Y)):
            Y[i] = X[i][0] * X[i][1]
        return X,Y