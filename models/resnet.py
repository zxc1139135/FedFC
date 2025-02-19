import os, sys

sys.path.append(os.pardir)

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.functional import softmax


def weights_init_ones(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def weights_init_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=1., std=1e-1)
        init.normal_(m.bias, mean=1., std=1e-1)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


import torch, torchvision


class InputNorm(nn.Module):
    def __init__(self, num_channel, num_feature_h, num_feature_w, dataTransMode):
        super().__init__()
        assert num_channel == 3, "only num_channel 3 is supported in InputNorm now"
        self.num_channel = num_channel
        # self.gamma = nn.Parameter(torch.ones(num_channel))
        # self.gamma = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(num_channel,)))
        # self.beta = nn.Parameter(torch.zeros(num_channel, num_feature_h, num_feature_w))
        # self.beta = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(num_channel, num_feature_h, num_feature_w)))
        # self.gamma = nn.Parameter(torch.ones(10))
        # self.beta = nn.Parameter(torch.zeros(10))
        # 设置参数范围
        self.min_val = -0.1
        self.max_val = 0.1
        # d
        self.gamma = nn.Parameter((self.max_val - self.min_val) * torch.rand(num_channel) + self.min_val)
        # nnd
        # self.gamma = nn.Parameter(
        #     (self.max_val - self.min_val) * torch.rand(num_channel, num_feature_h, num_feature_w) + self.min_val)
        # d
        # self.beta = nn.Parameter((self.max_val - self.min_val) * torch.rand(num_channel) + self.min_val)
        # nnd
        self.beta = nn.Parameter(
            (self.max_val - self.min_val) * torch.rand(num_channel, num_feature_h, num_feature_w) + self.min_val)
        # self.c = nn.Parameter(
        #     (self.max_val - self.min_val) * torch.rand(num_channel, num_feature_h, num_feature_w) + self.min_val)
        self.dataTransMode = dataTransMode

    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma * x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            assert self.dataTransMode in ['ax+b', 'relu(ax+b)', 'ax2+b', 'sigmoid(ax+b)', 'tanh(ax+b)', 'sigmoid(ax2+b)'
                                          , 'tanh(ax2+b)', '1c'], ("Your dataTransMode is not supported in \
                                                                           InputNorm now!")
            if self.dataTransMode == 'ax+b':
                return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta
            elif self.dataTransMode == 'relu(ax+b)':
                return F.relu(torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta)
            elif self.dataTransMode == 'sigmoid(ax+b)':
                return F.sigmoid(torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta)
            elif self.dataTransMode == 'tanh(ax+b)':
                return F.tanh(torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta)
            elif self.dataTransMode == 'ax2+b':
                # print('000000000000000000000000000000000000000000')
                # 1 d
                # beta_mean = self.beta.mean(dim=(1, 2), keepdim=True)
                # beta_expanded = beta_mean.expand_as(x)
                # return x ** 2 + beta_expanded
                # 1 nnd
                # return x ** 2 + self.beta
                # d nnd
                return torch.einsum('...ijk, i->...ijk', x, self.gamma)**2 + self.beta
                # d d
                # beta_mean = self.beta.mean(dim=(1, 2), keepdim=True)
                # beta_expanded = beta_mean.expand_as(x)
                # return torch.einsum('...ijk, i->...ijk', x, self.gamma) ** 2 + beta_expanded
                # d 0
                # return torch.einsum('...ijk, i->...ijk', x, self.gamma) ** 2
                # nnd 0
                # return torch.einsum('...ijk, ijk->...ijk', x, self.gamma) ** 2
                # nnd nnd
                # return torch.einsum('...ijk, ijk->...ijk', x, self.gamma) ** 2 + self.beta
                # nnd d
                # beta_mean = self.beta.mean(dim=(1, 2), keepdim=True)
                # beta_expanded = beta_mean.expand_as(x)
                # return torch.einsum('...ijk, ijk->...ijk', x, self.gamma) ** 2 + beta_expanded
            # elif self.dataTransMode == 'sigmoid(ax2+b)':
            #     return F.sigmoid(torch.einsum('...ijk, i->...ijk', x, self.gamma)**2 + self.beta)
            # elif self.dataTransMode == 'tanh(ax2+b)':
            #     return F.tanh(torch.einsum('...ijk, i->...ijk', x, self.gamma)**2 + self.beta)
            # elif self.dataTransMode == '1c':
                # print('test')
                # return torch.einsum('...ijk, i->...ijk', x, self.gamma)**2 + torch.einsum('...ijk, i->...ijk', x, self.beta) + self.c


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10, dataTrans=False, partys=2, dataTransMode='ax+b'):
        super().__init__()
        # net = torchvision.models.resnet18(pretrained=True)
        # n_ftrs = self.backbone.fc.in_features
        # self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)

        # 修改模型
        net = torchvision.models.resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
        net.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
        self.backbone = net

        # self.in_channels = 64
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(inplace=True))
        # # we use a different inputsize than the original paper
        # # so conv2_x's stride is 1
        # self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        # self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        # self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        # self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dataTrans = False
        if dataTrans == True:
            self.dataTrans = True
            self.dataTransMode = dataTransMode
            if num_classes == 8:
                self.norm = InputNorm(3, 16, 32, self.dataTransMode)
            else:
                # self.norm = InputNorm(3, 120)
                if partys == 2:
                    self.norm = InputNorm(3, 16, 32, self.dataTransMode)
                elif partys == 4:
                    self.norm = InputNorm(3, 16, 16, self.dataTransMode)
                else:
                    assert (partys == 2 or partys == 4), "In resnet18, total number of parties not supported for data partitioning"

    # def _make_layer(self, block, out_channels, num_blocks, stride):
    #     """make resnet layers(by layer i didnt mean this 'layer' was the
    #     same as a neuron netowork layer, ex. conv layer), one layer may
    #     contain more than one residual block
    #     Args:
    #         block: block type, basic block or bottle neck block
    #         out_channels: output depth channel number of this layer
    #         num_blocks: how many blocks per layer
    #         stride: the stride of the first block of this layer
    #     Return:
    #         return a resnet layer
    #     """
    #
    #     # we have num_block blocks per layer, the first block
    #     # could be 1 or 2, other blocks would always be 1
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels * block.expansion
    #
    #     return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataTrans:
            x = self.norm(x)
            # print("self.norm里的alpha和beta", self.norm.gamma, self.norm.beta)
        # output = self.conv1(x)
        # output = self.conv2_x(output)
        # output = self.conv3_x(output)
        # output = self.conv4_x(output)
        # output = self.conv5_x(output)
        # output = self.avg_pool(output)
        # output = self.dropout(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        # return output

        logits = self.backbone(x)
        # if self.dataTrans:
        #     output = self.norm(output)
        return logits
        # return softmax(logits, dim=-1)


def resnet18(num_classes=10, dataTrans=False, partys=0, dataTransMode='ax+b'):
    """ return a ResNet 18 object

    Args:
        dataTrans: 是否dataTrans
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dataTrans, partys, dataTransMode)


def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


# for resnet20
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(num_classes=10):
    kernel_size = (3, 3)
    return ResNet2(block=BasicBlock2, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet110(num_classes=10):
    kernel_size = (3, 3)
    return ResNet2(block=BasicBlock2, num_blocks=[18, 18, 18], kernel_size=kernel_size, num_classes=num_classes)


def resnet56(num_classes=10):
    kernel_size = (3, 3)
    return ResNet2(block=BasicBlock2, num_blocks=[9, 9, 9], kernel_size=kernel_size, num_classes=num_classes)
