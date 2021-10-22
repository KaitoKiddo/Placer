import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from layers import GraphConvolution
class Actor(nn.Module):

    #GCN
    def __init__(self, nfeat, nhid, relu=True, bn=False):
        super(Actor, self).__init__()
        self.gc1 = GraphConvolution(2, 2)
        self.fc1 = nn.Linear(90, 180)
        self.gc2 = GraphConvolution(4, 8)
        self.fc2 = nn.Linear(360, 512)
        self.dropout = 0.5
    #五层Deconv
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1) #8X8
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1) #16X16
        self.conv3 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1) #32X32
        self.conv4 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1) #64X64
        self.conv5 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1) #128X128
        #Hout =（H-1）St-2p+k
        self.bn = nn.BatchNorm2d(out_channels=1, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x, adj):
        # x = torch.reshape(x,(1,90))
        # print(x.shape)
        x = self.gc1(x, adj)
        x = torch.reshape(x, (1, 90))
        x = self.fc1(x)
        x = torch.reshape(x, (45, 4))
        # print(x.shape)
        x = self.gc2(x, adj)
        x = torch.reshape(x, (1, 360))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        x = torch.reshape(x, (1, 32, 4, 4))
    #deconv forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x)
        x = torch.reshape(x, (1, 128*128))
        x = x.squeeze(1)
        # x = torch.squeeze(x,1).float()
        # print(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
    #根据网表的大小设计Deconv层数和参数
        dist = Categorical(x)
        # print(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, nfeat, nhid, relu=True, bn=False):
        super(Critic, self).__init__()
        self.gc1 = GraphConvolution(2, 2)
        self.fc1 = nn.Linear(90, 180)
        self.gc2 = GraphConvolution(4, 8)
        self.fc2 = nn.Linear(360, 1)
        self.dropout = 0.5

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.reshape(x, (1, 90))
        x = self.fc1(x)
        x = torch.reshape(x, (45, 4))
        x = self.gc2(x, adj)
        x = torch.reshape(x, (1, 360))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # deconv forward
        # x = x.view(x.size(0), -1)
        # value = self.linear(x)
        return x


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
#         # 经过处理后的x要与x的维度相同(尺寸和深度)
#         # 如果不相同，需要添加卷积+BN来变换为同一维度
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
