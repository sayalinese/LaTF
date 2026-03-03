import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear
from collections import OrderedDict


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


import numpy as np
import torch
from torch import nn


# from models.containers import Module


class ScaledDotProductWithMapAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_in, d_out, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithMapAttention, self).__init__()
        self.d_in = d_in
        self.fc_q = nn.Linear(d_in, h * d_k)
        self.fc_k = nn.Linear(d_in, h * d_k)
        self.fc_v = nn.Linear(d_in, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_out)
        self.dropout = nn.Dropout(dropout)

        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, loss_map, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = loss_map.permute(0, 2, 1).unsqueeze(2)  # bs * 8 * 1 * 197
        w_a = att

        # w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        # print(w_a.shape)
        # print(w_g.shape)
        w_mn = torch.softmax(w_a + w_g, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ScaledDotProductWithMapAttentionV2(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_in, d_out, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithMapAttentionV2, self).__init__()
        self.d_in = d_in
        self.fc_q = nn.Linear(d_in, h * d_k)
        self.fc_k1 = nn.Linear(d_in, h * d_k)
        self.fc_k2 = nn.Linear(d_in, h * d_k)
        self.fc_v = nn.Linear(d_in, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_out)
        self.dropout = nn.Dropout(dropout)

        self.relative_importance_mapping = nn.Linear(d_in, h)

        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k1.weight)
        nn.init.xavier_uniform_(self.fc_k2.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k1.bias, 0)
        nn.init.constant_(self.fc_k2.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys1, keys2, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys1.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k1 = self.fc_k1(keys1).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        k2 = self.fc_k2(keys2).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # relative_k1 = self.relative_importance_mapping(keys1).view(b_s, nk, self.h, 1)  # (b_s, nk, h, 1)
        # relative_k2 = self.relative_importance_mapping(keys2).view(b_s, nk, self.h, 1)  # (b_s, nk, h, 1)
        # k1k2 = torch.cat([relative_k1, relative_k2], dim=-1)  # (b_s, nk, h, 2)
        # att_k1k2 = torch.softmax(k1k2, dim=-1)  # (b_s, nk, h, 2)
        # k1_importance = att_k1k2[:, :, :, 0].contiguous().permute(0, 2, 1).unsqueeze(2)  # (b_s, h, 1, nk)
        # k2_importance = att_k1k2[:, :, :, 1].contiguous().permute(0, 2, 1).unsqueeze(2)  # (b_s, h, 1, nk)

        att = (torch.matmul(q, k1) + torch.matmul(q, k2)) / np.sqrt(
            self.d_k * 2)  # (b_s, h, nq, nk)
        
        # 数值稳定性: clamp 防止溢出
        att = torch.clamp(att, min=-50, max=50)
        
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -1e9)  # 使用 -1e9 而不是 -inf

        # w_g = loss_map.permute(0, 2, 1).unsqueeze(2)  # bs * 8 * 1 * 197
        w_a = att

        # 数值稳定的 softmax
        w_a_max = w_a.max(dim=-1, keepdim=True)[0]
        w_mn = torch.softmax(w_a - w_a_max, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadMapAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_in=1024, d_out=1024, d_k=64, d_v=64, h=8, dropout=.1, identity_map_reordering=False,
                 spacial_dim=14):
        super(MultiHeadMapAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithMapAttention(d_in=d_in, d_out=d_out, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)  # 197 * d_in

    def forward(self, feature_map, loss_map, attention_mask=None, attention_weights=None):
        # feature_map bs * 1024 * 14 * 14
        # loss_map bs * 8 * 14 * 14
        x = feature_map.flatten(start_dim=2)  # bs * 1024 * 196
        x = x.permute(0, 2, 1)  # bs * 196 * 1024
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # bs * 197 * 1024

        loss_map = loss_map.flatten(start_dim=2)  # bs * 8 * 196
        loss_map = loss_map.permute(0, 2, 1)  # bs * 196 * 8
        loss_map = torch.cat([loss_map.mean(dim=1, keepdim=True), loss_map], dim=1)  # bs * 197 * 8

        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # bs * 197 * 1024
        queries = x[:, :1]
        keys = x
        values = x
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, loss_map, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, loss_map, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out.squeeze()


class MultiHeadMapAttentionV2(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_in=1024, d_out=1024, d_k=64, d_v=64, h=8, dropout=.1, identity_map_reordering=False,
                 spacial_dim=14):
        super(MultiHeadMapAttentionV2, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention1 = ScaledDotProductWithMapAttentionV2(d_in=d_in, d_out=d_out, d_k=d_k, d_v=d_v, h=h)
        self.attention2 = ScaledDotProductWithMapAttentionV2(d_in=d_in, d_out=d_out, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_out)
        self.layer_norm2 = nn.LayerNorm(d_out)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)  # 197 * d_in

    def forward(self, feature_map, loss_map, attention_mask=None, attention_weights=None):
        # feature_map bs * 1024 * 14 * 14
        # loss_map bs * 8 * 14 * 14
        x = feature_map.flatten(start_dim=2)  # bs * 1024 * 196
        x = x.permute(0, 2, 1)  # bs * 196 * 1024
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # bs * 197 * 1024
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # bs * 197 * 1024

        loss_map = loss_map.flatten(start_dim=2)  # bs * 1024 * 196
        loss_map = loss_map.permute(0, 2, 1)  # bs * 196 * 1024
        loss_map = torch.cat([loss_map.mean(dim=1, keepdim=True), loss_map], dim=1)  # bs * 197 * 1024
        loss_map = loss_map + self.positional_embedding[None, :, :].to(loss_map.dtype)

        queries1 = x[:, :1]
        # queries = torch.cat([x[:, :1], loss_map[:, :1]], dim=1)  # bs * 2 * 1024
        keys1 = x
        keys2 = loss_map
        values1 = x

        out1 = self.attention1(queries1, keys1, keys2, values1, attention_mask, attention_weights)
        out1 = self.dropout(out1)
        out1 = self.layer_norm1(queries1 + out1)

        queries2 = loss_map[:, :1]
        values2 = loss_map

        out2 = self.attention2(queries2, keys1, keys2, values2, attention_mask, attention_weights)
        out2 = self.dropout(out2)
        out2 = self.layer_norm2(queries2 + out2)

        out = torch.cat([out1.squeeze(1), out2.squeeze(1)], dim=-1)  # bs * 4096
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(4, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
