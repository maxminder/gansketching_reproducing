from jittor.utils.pytorch_converter import convert
import numpy as np
#import torch
import jittor as jt
#from torch import jit
#from torch._C import device
class YourDataset(jt.dataset.Dataset):
    def __init__(self):
        super().__init__(batch_size=32)
        self.set_attrs(total_len=100)

    def __getitem__(self, k):
        return k, k*k

x = jt.randn((2,3))
x = x.contiguous()