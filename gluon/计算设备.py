import mxnet as mx
from mxnet import nd
import numpy as np
a=nd.arange(1,13).reshape(3,4)
b=a.as_in_context(mx.gpu())
c=nd.array(np.arange(1,13).reshape(3,4),ctx=mx.gpu())
d=nd.arange(1,13,ctx=mx.gpu()).reshape((3,4))
print(a,b,c,d,sep='\n')