from mxnet import nd
import numpy as np
a=np.arange(12)
x=nd.arange(12)
x.attach_grad()
print(x)
print(x.shape,x.size)