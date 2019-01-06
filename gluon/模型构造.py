from mxnet.gluon import nn
from mxnet import init
from mxnet import nd
import mxnet as mx


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()


class AlexNet(nn.HybridBlock):
    def __init__(self, classes=1000, verbose=False, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                # Flatten层负责将(N,C,H,W)维度变为(N,C*H*W)维度
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        for i, b in enumerate(self.features):
            x = b(x)
            if self.verbose:
                print("features[%d] 输出维度: %s" % (i + 1, x.shape))
        x = self.output(x)
        if self.verbose:
            print("output 输出维度: %s" % (x.shape,))
        return x


def get_net(ctx, verbose=False):
    classes = 1000
    net = AlexNet(classes, verbose=verbose)
    # 定义的网络需要初始化设备和参数
    # 相比于pytorch，gluon不需要指定输入的通道数，就是由于先对网络进行了初始化，
    # 可以推断出来输入的通道数量
    net.initialize(ctx=ctx, init=init.Xavier())
    net.hybridize()
    # net.initialize()
    # 默认的初始化方法是把所有权重初始化成在[-0.07, 0.07]之间均匀分布的随机数
    return net


if __name__ == '__main__':
    x = nd.random.uniform(shape=(32, 3, 227, 227), ctx=ctx)
    net = get_net(ctx, verbose=False)
    print(net)
    y = net(x)
    # print("输出y的维度: %s" % (y.shape,))
    print(y)
