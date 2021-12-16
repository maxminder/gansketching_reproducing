#from jittor.utils.pytorch_converter import convert
import numpy as np
import torch
#import jittor as jt
#from torch._C import device
stddev = torch.randn(2,3,4,5)
out = stddev.var(0,unbiased=False)
stddev = stddev - stddev.mean(0,keepdim=True)
stddev = stddev.square()
stddev = stddev.sum(0) / stddev.shape[0]
print(stddev)
print(out)