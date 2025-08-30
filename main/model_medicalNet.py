import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4):
        super(MultiHeadCrossAttention, self).__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.query_proj = nn.Conv3d(query_dim, query_dim, kernel_size=1)
        self.key_proj = nn.Conv3d(context_dim, query_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(context_dim, query_dim, kernel_size=1)
        self.out_proj = nn.Conv3d(query_dim, query_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        B, C, D, H, W = x.shape
        N = D * H * W

        # Project to query, key, value
        Q = self.query_proj(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, dim]
        K = self.key_proj(context).view(B, self.num_heads, self.head_dim, N)                  # [B, heads, dim, N]
        V = self.value_proj(context).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, dim]

        # Attention
        attn_weights = torch.matmul(Q, K) / math.sqrt(self.head_dim)  # [B, heads, N, N]
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, N, dim]

        # Concatenate heads
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(B, C, D, H, W)
        out = self.out_proj(attn_output)

        # Residual
        return self.gamma * out + x
    

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SelfAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)  # shape: [B, C]
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)
    

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv3d(query_dim, query_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(context_dim, query_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(context_dim, query_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        B, C, D, H, W = x.size()  # x 是 query
        proj_query = self.query_conv(x).view(B, -1, D * H * W).permute(0, 2, 1)   # [B, N, C_q]
        proj_key = self.key_conv(context).view(B, -1, D * H * W)                  # [B, C_k, N]
        energy = torch.bmm(proj_query, proj_key)                                 # [B, N, N]
        attention = self.softmax(energy)
        proj_value = self.value_conv(context).view(B, -1, D * H * W)             # [B, C_v, N]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                  # [B, C_v, N]
        out = out.view(B, C, D, H, W)
        out = self.gamma * out + x
        return out


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 in_channels=1,
                 no_cuda = False
                 ):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, # 你實際輸入資料的 channel 數
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)

        self.attn3 = SelfAttention3D(256 * block.expansion)  # 加在 layer3 後
        self.attn4 = SelfAttention3D(512 * block.expansion)


        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        # self.cross_attention = CrossAttention(query_dim=1024, context_dim=512)
        self.cross_attention = MultiHeadCrossAttention(query_dim=1024, context_dim=512, num_heads=4)



        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.3)  # 加在全連接層前

        self.fc = nn.Linear(512 * block.expansion, num_seg_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)         # Cross Attention 的 context
        x3 = self.layer3(x2)        # Cross Attention 的 query
        x3 = self.attn3(x3)  # 🔥 Apply self-attention here
        # x3 = self.cross_attention(x3, x2)
        x = self.layer4(x3)
        # x = self.attn4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model