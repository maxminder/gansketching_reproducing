import numpy as np
a = np.random.rand(2,3)
b = np.random.rand(2,5)
a = np.concatenate((a,b),axis=1)
print(a.shape)