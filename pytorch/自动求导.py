import torch
import numpy as np
x=torch.tensor(np.arange(1,13).reshape(3,4),dtype=torch.float32,requires_grad=True)
print(x)
y=x**2+4*x
z=2*y+3
print(z)
z.sum().backward()
print(x.grad)
