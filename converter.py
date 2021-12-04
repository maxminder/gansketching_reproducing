#from jittor.utils.pytorch_converter import convert
import numpy as np
#import torch
import jittor as jt
#from torch._C import device
inputs = jt.randn(1, 4, 5, 5)
weights = jt.randn(1, 9, 3, 3)
a = jt.misc.split(inputs,1,dim=1)
result = []
for b in a:
    result.append(jt.nn.conv_transpose2d(b,weights,padding=1))
x = jt.concat(result,dim=1)
print(x.shape)
#x = jt.nn.conv_transpose2d(inputs, weights, padding=1)