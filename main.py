from function import function
from function import sigmoid

from layer import layer
from rnm import rnm
import numpy as np

from data import data

inputs = 2
outputs = 1
epochs = 1500
batch_size = 35
lr = 0.01
sample_test = 1000
range_a = -1.0
range_b = 1.0
n_tests = 10

NN = rnm.RNM(inputs,outputs)

d = data.DATA()

X,Y = d.generate_data_set_f_xy(sample_test,range_a,range_b)
print(X.shape)
print(Y.shape)


NN.train(X,Y,epochs,batch_size,lr)

X,Y = d.generate_data_set_f_xy(n_tests,range_a,range_b)

test1 = [0.5, 0.5]
test2 = [0.25, -0.25]
test3 = [0.0, 0.0]
test4 = [1, -1]

# test5 = [0.0, 0.0]
# test6 = [0.0, 0.0]

pred1,_ = NN.predict(test1)
pred2,_ = NN.predict(test2)
pred3,_ = NN.predict(test3)
pred4,_ = NN.predict(test4)
# pred5,_ = NN.predict(test5)
# pred6,_ = NN.predict(test6)
    