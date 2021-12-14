#from jittor.utils.pytorch_converter import convert
import torch
import jittor as jt
import numpy as np
#from torch._C import device
# inputs = torch.randn(1, 4, 5, 5)
# weights = torch.randn(4, 9, 3, 3)
# a = torch.split(inputs,1,dim=1)
# b = torch.split(weights,1,dim=0)
# result = []
# for i in range(len(a)):
#     result.append(torch.conv_transpose2d(a[i],b[i],padding=1))
# x = torch.concat(result,dim=1)
# y = torch.conv_transpose2d(inputs,weights,padding=1,groups=4)
# print(x==y)
#x = jt.nn.conv_transpose2d(inputs, weights, padding=1)
a = jt.ones((2,3))
a = np.array(a)
print(a)