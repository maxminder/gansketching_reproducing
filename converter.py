#from jittor.utils.pytorch_converter import convert
import numpy as np
import torch
#import jittor as jt
#from torch._C import device

x = torch.randn((2,3,4,5))
print(x.new_empty(0))
#print(x)