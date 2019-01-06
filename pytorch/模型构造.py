import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000,verbose=False):
        super(AlexNet, self).__init__()
        self.verbose=verbose
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        for i, b in enumerate(self.features):
            x = b(x)
            if self.verbose:
                print("features[%d] 输出维度: %s" % (i + 1, x.shape))
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        for i,b in enumerate(self.classifier):
            x=b(x)
            if self.verbose:
                print("classifier[%d] 输出维度: %s" % (i + 1, x.shape))
        return x

def get_net(device,verbose=False):
    classes=1000
    net=AlexNet(classes,verbose=True).to(device)
    # net = AlexNet(classes, verbose=True,device=device)

    return net

if __name__=='__main__':
    x=torch.nn.init.uniform_(torch.Tensor(32,3,227,227)).to(device)
    net=get_net(device,verbose=True)
    print(net)
    y=net(x)
    print("输出y的维度: %s" % (y.shape,))
    optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=wd)

