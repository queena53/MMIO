import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Any


class SelfAttention3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape 3D tensor to 2D for attention
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        x = x.permute(0, 2, 1).view(b, c, d, h, w)  # Reshape back to 3D
        return x


class _DenseLayer3D(nn.Module):
    def __init__(self, input_c, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(input_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(input_c, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, input_c, bn_size, growth_rate, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer3D(input_c + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition3D(nn.Sequential):
    def __init__(self, input_c, output_c):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(input_c, output_c, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm3d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate

            # Add Self-Attention after each DenseBlock
            self.features.add_module(f"self_attention{i + 1}", SelfAttention3D(num_features))

            if i != len(block_config) - 1:
                trans = _Transition3D(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out