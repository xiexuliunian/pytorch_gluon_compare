from mxnet import autograd,nd
x=nd.arange(1,13).reshape((3,4))
print(x)
x.attach_grad()
with autograd.record():
    y=x**2+4*x
    z=2*y+3
print(z)
z.backward()
print(x.grad)