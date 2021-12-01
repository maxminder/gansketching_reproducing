#from jittor.utils.pytorch_converter import convert
import numpy as np
#import torch
import jittor as jt
#from torch._C import device
jt.dirty_fix_pytorch_runtime_error()
import torch
a = torch.randn(2,3)
b = jt.mean(a)
print(b)
#print(x)