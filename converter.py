#from jittor.utils.pytorch_converter import convert
import numpy as np
import torch
#import jittor as jt
#from torch._C import device
stddev = torch.Tensor([[1,2,3],[2,3,4]])
stddev = stddev - stddev.mean(1,keepdim=True)
print(stddev.mean(1).shape)
print(stddev)