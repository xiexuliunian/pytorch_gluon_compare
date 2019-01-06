import torch
import numpy as np
a=np.arange(12)
b=torch.from_numpy(a)
x=torch.arange(12)
print(x)
print(x.shape,x.size())
print(b)