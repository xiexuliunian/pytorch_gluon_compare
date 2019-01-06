import torch
import numpy as np
a=torch.tensor(np.arange(1,13).reshape(3,4),device=torch.device('cuda'))
b=torch.tensor(np.arange(1,13).reshape(3,4)).cuda()
c=torch.arange(1,13,device=torch.device('cuda')).reshape(3,4)
d=torch.arange(1,13).cuda().reshape(3,4)
e=torch.arange(1,13).reshape(3,4).cuda()
print(a,b,c,d,e,sep='\n')