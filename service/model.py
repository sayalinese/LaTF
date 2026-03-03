import os
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear
from torchvision.models.resnet import resnet50
from service.model_base import MultiHeadMapAttention, MultiHeadMapAttentionV2, ResNet, ResidualBlock
from service.feature_extractors import MultiFeatureExtractor

# 使用原版 OpenAI CLIP (V4-V7 需要)
import clip

# 模型类型映射，无需本地文件
type_to_path = {
    'RN50x64': 'RN50x64',  # open_clip会自动下载
    'RN50': 'RN50',
    'RN50x4': 'RN50x4',
    'RN50x16': 'RN50x16',
    'ViT-B/32': 'ViT-B/32',
    'ViT-B/16': 'ViT-B/16',
    'ViT-L/14': 'ViT-L/14',
}


class Res50Feature(nn.Module):
    def __init__(self, num_class=4):
        super(Res50Feature, self).__init__()
        # 使用 torchvision 的预训练权重 (会自动下载)
        from torchvision.models import ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print('ResNet50 loaded with ImageNet weights')
        self.model.fc = nn.Identity()

    def forward(self, image_input, isTrain=False):
        image_feats = self.model(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return None, image_feats


class LASTED(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(LASTED, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.output_layer = Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.GELU(),
        #     nn.Linear(1280, 512),
        # )
        # self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, isTrain=True):
        if isTrain:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


class CLipClassifier(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifier, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV2, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input, loss_map=''):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV3(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV3, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        self.clip_model.visual.attnpool = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, image_input, loss_map=''):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = self.pool(image_feats)
        image_feats = torch.flatten(image_feats, 1)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV4(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV4, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class ResnetBaseline(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(ResnetBaseline, self).__init__()
        # 使用 torchvision 的预训练权重 (会自动下载)
        from torchvision.models import ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(2048, 2)

    def forward(self, image_input, loss_map):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.model(image_input)  # bs * 2048
        logits = self.fc(image_feats)
        return logits


class CLipClassifierWMap(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMap, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        # image_feats_attnpooled_normed = image_feats_attnpooled
        # print('init:', loss_map.shape)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        # print('conv:', loss_map.shape)
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        # feats_with_weights_pooled = F.adaptive_avg_pool2d(feats_with_weights, (1, 1)).squeeze() # bs * 2048
        image_feats = feats_with_weights
        image_feats_attnpooled = self.visual_attnpool(image_feats)

        final_feats = image_feats_attnpooled
        # print(image_feats.shape)
        # print(loss_map.shape)
        logits = self.fc(final_feats)
        return logits


class CLipClassifierWMapV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV2, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        # image_feats_attnpooled_normed = image_feats_attnpooled
        # print('init:', loss_map.shape)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        # print('conv:', loss_map.shape)
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        image_feats_attnpooled = self.visual_attnpool(image_feats + feats_with_weights)
        logits = self.fc(image_feats_attnpooled)
        return logits


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class CLipClassifierWMapV3(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV3, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(512, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.up_feature_map = Upsample(2048, 512, with_conv=True)
        self.down_loss_map = Downsample(4, 1, with_conv=True)

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats = self.clip_model.encode_image(image_input)  # bs * 2048 * 7 * 7
        up_image_feats = self.up_feature_map(image_feats)  # bs * 512 * 14 * 14

        down_loss_map = self.down_loss_map(loss_map)  # bs * 16 * 16 * 16
        down_loss_map = F.adaptive_avg_pool2d(down_loss_map, (14, 14))  # bs * 1 * 14 * 14

        up_image_feats = up_image_feats * torch.sigmoid(down_loss_map)
        global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512

        logits = self.fc(global_feats)
        return logits


def forward(self, x):
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x_l3 = self.layer3(x)
    x = self.layer4(x_l3)
    x = self.attnpool(x)

    return x, x_l3


def encode_image(self, image):
    """Wrapper to call visual.forward (which is set to forward function)"""
    x, x_l3 = self.visual(image.type(self.dtype))
    return x, x_l3


class CLipClassifierWMapV4(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV4, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024, 2)
        # self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.forward = forward
        self.clip_model.encode_image = encode_image

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(self.clip_model, image_input)
        # block3 bs * 1024 * 14 * 14
        # print(block3_feats.shape)
        # up_image_feats = self.up_feature_map(image_feats)  # bs * 1024 * 14 * 14

        loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(loss_map)  # bs * 1 * 14 * 14
        up_image_feats = block3_feats * torch.sigmoid(pooled_loss_map)
        global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512

        logits = self.fc(torch.cat([image_feats, global_feats], dim=1))
        return logits


class CLipClassifierWMapV5(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV5, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024, 2)
        # self.visual_attnpool = self.clip_model.visual.attnpool
        # Monkey patch
        self.clip_model.visual.forward = forward
        self.clip_model.encode_image = encode_image

        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.attn_pool = MultiHeadMapAttention()

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(self.clip_model, image_input)
        # block3 bs * 1024 * 14 * 14
        # print(block3_feats.shape)
        # up_image_feats = self.up_feature_map(image_feats)  # bs * 1024 * 14 * 14

        loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(loss_map)  # bs * 8 * 14 * 14

        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # bs * 1024
        # print(pooled_block3_feats.shape)
        # up_image_feats = block3_feats * torch.sigmoid(pooled_loss_map)
        # global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512
        logits = self.fc(torch.cat([image_feats, pooled_block3_feats], dim=1))
        return logits


class CLipClassifierWMapV6(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV6, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024 + 1024, 2)
        # self.visual_attnpool = self.clip_model.visual.attnpool
        # Monkey patch
        self.clip_model.visual.forward = forward
        self.clip_model.encode_image = encode_image

        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        self.attn_pool = MultiHeadMapAttention()
        self.channel_align = ChannelAlignLayer(4, 128, 1024)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(self.clip_model, image_input)
        # block3 bs * 1024 * 14 * 14
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(aligned_loss_map)  # bs * 8 * 14 * 14
        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # bs * 1024

        channel_weighted_feats = self.channel_align(block3_feats, loss_map)  # bs * 1024
        logits = self.fc(torch.cat([image_feats, pooled_block3_feats, channel_weighted_feats], dim=1))
        return logits


class CLipClassifierOnlyMapV1(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierOnlyMapV1, self).__init__()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.model = ResNet(ResidualBlock, [2, 2, 2], 2)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        logits = self.model(loss_map)
        return logits


class CLipClassifierOnlyMapV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierOnlyMapV2, self).__init__()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.model = ResNet(ResidualBlock, [2, 2, 2], 2)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        logits = self.model(loss_map)
        return logits


class CLipClassifierWMapV7(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV7, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        
        # RN50x64: image_feats=1024, block3_feats=2048*28*28
        # 最终特征: 1024 (image) + 4096 (attn) + 2048 (channel) = 7168
        self.fc = nn.Linear(1024 + 4096 + 2048, 2)
        
        # Monkey patch to get intermediate features (use types.MethodType for correct binding)
        self.clip_model.visual.forward = types.MethodType(forward, self.clip_model.visual)
        self.clip_model.encode_image = types.MethodType(encode_image, self.clip_model)

        # 匹配 RN50x64 的 block3 维度 (2048)
        self.conv = nn.Conv2d(4, 2048, kernel_size=(1, 1))
        self.channel_align = ChannelAlignLayer(4, 256, 2048)
        # spacial_dim=28 匹配 RN50x64 的空间维度
        self.attn_pool = MultiHeadMapAttentionV2(d_in=2048, d_out=2048, spacial_dim=28)
        
        # 初始化新增层的权重
        self._init_weights()
    
    def _init_weights(self):
        # Xavier 初始化
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, image_input, loss_map):
        # loss_map: bs * 4 * 32 * 32
        # 直接调用修改后的 visual forward (which returns x, x_l3)
        with torch.no_grad():
            self.clip_model.eval()
        result = self.clip_model.visual.forward(image_input)
        if isinstance(result, tuple) and len(result) == 2:
            image_feats, block3_feats = result
        else:
            # 如果 forward 只返回单个值，需要处理
            image_feats = result
            block3_feats = None
        # image_feats: bs * 1024
        # block3_feats: bs * 2048 * 28 * 28
        
        # 对齐 loss_map 到 block3 的空间维度
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (28, 28))  # bs * 4 * 28 * 28
        upscaled_loss_map = self.conv(aligned_loss_map)  # bs * 2048 * 28 * 28

        # 多头注意力融合 (输出 4096 = 2048*2)
        refined_feats = self.attn_pool(block3_feats, upscaled_loss_map)

        # 通道对齐 (输出 2048)
        channel_weighted_feats = self.channel_align(block3_feats, loss_map)
        
        # 拼接所有特征并分类
        all_feats = torch.cat([image_feats, refined_feats, channel_weighted_feats], dim=1)
        logits = self.fc(all_feats)
        return logits


class PoolWithMap(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, loss_map):
        # x bs * 1024 * 14 * 14
        # loss_map bs * 1024 * 14 * 14
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        loss_map = loss_map.flatten(start_dim=2).permute(2, 0, 1)
        loss_map = torch.cat([loss_map.mean(dim=0, keepdim=True), loss_map], dim=0)
        loss_map = loss_map + self.positional_embedding[:, None, :].to(loss_map.dtype)  # (HW+1)NC

        queries = torch.cat([x[:1], loss_map[:1]], dim=0)  # 2 * N * C

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


class ChannelAlignLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ChannelAlignLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map, loss_map):
        b, c, _, _ = feature_map.size()
        pooled_feature_map = self.avg_pool(feature_map).view(b, c)  # bs * 2048
        pooled_loss_map = self.avg_pool(loss_map).view(b, -1)  # bs * 4
        weights = self.fc(pooled_loss_map)
        out = pooled_feature_map * weights
        return out


class CLipClassifierWMapBaseline(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapBaseline, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(2048, 2)
        self.clip_model.visual.attnpool = nn.Identity()
        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        final_feats = F.adaptive_avg_pool2d(feats_with_weights, (1, 1)).squeeze()
        logits = self.fc(final_feats)
        return logits


class LASTEDWLoss(nn.Module):
    def __init__(self, num_class=4):
        super(LASTEDWLoss, self).__init__()
        self.clip_model, self.preprocess = clip.load('RN50x64', device='cpu', jit=False)
        # self.output_layer = Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.GELU(),
        #     nn.Linear(1280, 512),
        # )
        # self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, isTrain=True):
        if isTrain:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


class CLipClassifierWMapV9Lite(nn.Module):
    """
    V9-Lite: Lightweight model for local tampering detection
    Optimized for doubao local inpainting with minimal params
    Integrated with TruFor reliability map mechanism
    """
    def __init__(self, num_class=2, clip_type="RN50x64", use_trufor=False):
        super(CLipClassifierWMapV9Lite, self).__init__()
        self.clip_model, _ = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        
        # Monkey patch to get intermediate features (use types.MethodType for correct binding)
        self.clip_model.visual.forward = types.MethodType(forward, self.clip_model.visual)
        self.clip_model.encode_image = types.MethodType(encode_image, self.clip_model)
        
        # Simplified: only use global features + channel alignment
        self.fc = nn.Linear(1024 + 2048, 2)
        
        # LaRE map processing with TOP-K pooling (for local anomaly)
        self.conv = nn.Conv2d(4, 2048, kernel_size=1)
        
        # Channel alignment with residual gating
        self.channel_align = ChannelAlignLayerV2(4, 2048)
        
        # Optional: localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # TruFor reliability map integration
        self.use_trufor = use_trufor
        if use_trufor:
            # Confidence head (parallel to localization head)
            self.conf_head = nn.Sequential(
                nn.Conv2d(2048, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 1, 1),
                nn.Sigmoid()
            )
            
            # Detection head for weighted statistics pooling (similar to TruFor)
            self.detection = nn.Sequential(
                nn.Linear(in_features=8, out_features=128),  # 4 stats * 2 maps (loc+conf)
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=128, out_features=1),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        if self.use_trufor:
            nn.init.xavier_uniform_(self.conf_head[0].weight)
            if self.conf_head[0].bias is not None:
                nn.init.zeros_(self.conf_head[0].bias)
            nn.init.xavier_uniform_(self.conf_head[2].weight)
            if self.conf_head[2].bias is not None:
                nn.init.zeros_(self.conf_head[2].bias)
            
            nn.init.xavier_uniform_(self.detection[0].weight)
            nn.init.zeros_(self.detection[0].bias)
            nn.init.xavier_uniform_(self.detection[3].weight)
            nn.init.zeros_(self.detection[3].bias)
    
    def forward(self, image_input, loss_map, return_loc=False, return_conf=False):
        """
        Args:
            image_input: [B, 3, 448, 448]
            loss_map: [B, 4, 32, 32]
            return_loc: whether to return localization map
            return_conf: whether to return confidence map (only when use_trufor=True)
        """
        # Get CLIP features (monkey patched visual returns (image_feats, block3_feats))
        with torch.no_grad():
            self.clip_model.eval()
            image_feats, block3_feats = self.clip_model.visual(image_input)
        
        if block3_feats is None:
            # Fallback: use dummy features
            b = image_input.size(0)
            block3_feats = torch.zeros(b, 2048, 28, 28, device=image_input.device)
        
        # Align loss_map to block3 spatial size
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (28, 28))
        upscaled_loss_map = self.conv(aligned_loss_map)
        
        # Channel alignment (with top-k pooling inside)
        channel_weighted = self.channel_align(block3_feats, loss_map)
        
        # Classification
        all_feats = torch.cat([image_feats, channel_weighted], dim=1)
        logits = self.fc(all_feats)
        
        # TruFor reliability map mechanism
        if self.use_trufor:
            # Generate localization and confidence maps
            loc_map = self.loc_head(upscaled_loss_map)
            conf_map = self.conf_head(upscaled_loss_map)
            
            # Weighted statistics pooling (TruFor style)
            f1 = weighted_statistics_pooling(conf_map).view(logits.shape[0], -1)
            f2 = weighted_statistics_pooling(loc_map, torch.log(conf_map + 1e-8)).view(logits.shape[0], -1)
            
            # Detection head for refined classification
            det_features = torch.cat((f1, f2), -1)
            det_logits = self.detection(det_features)
            
            # Combine original logits with detection logits
            combined_logits = (logits + det_logits) / 2
            
            outputs = [combined_logits]
            if return_loc:
                outputs.append(loc_map)
            if return_conf:
                outputs.append(conf_map)
            
            if len(outputs) == 1:
                return outputs[0]
            elif len(outputs) == 2:
                return outputs[0], outputs[1]
            else:
                return outputs[0], outputs[1], outputs[2]
        
        else:
            # Original behavior
            if return_loc:
                loc_map = self.loc_head(upscaled_loss_map)
                return logits, loc_map
            
            return logits


class ChannelAlignLayerV2(nn.Module):
    """
    Improved channel alignment with TOP-K pooling for local anomaly
    """
    def __init__(self, lare_channels=4, feature_channels=2048, topk_ratio=0.05):
        super(ChannelAlignLayerV2, self).__init__()
        self.topk_ratio = topk_ratio
        
        # FC to generate weights
        self.fc = nn.Sequential(
            nn.Linear(lare_channels, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Residual gating scale
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, feature_map, lare_features):
        """
        Args:
            feature_map: [B, C, H, W]
            lare_features: [B, 4, H', W']
        """
        b, c, h, w = feature_map.shape
        
        # Global avg pool for features
        pooled_feat = F.adaptive_avg_pool2d(feature_map, 1).view(b, c)
        
        # TOP-K pooling for lare (capture local peaks)
        lare_flat = lare_features.view(b, lare_features.size(1), -1)
        k = max(1, int(lare_flat.size(-1) * self.topk_ratio))
        topk_vals, _ = torch.topk(lare_flat, k=k, dim=-1)
        pooled_lare = topk_vals.mean(dim=-1)  # [B, 4]
        
        # Generate channel weights
        weights = self.fc(pooled_lare)  # [B, C]
        
        # Residual gating: out = feat * (1 + alpha * weight)
        out = pooled_feat * (1 + self.alpha * weights)
        
        return out


def weighted_statistics_pooling(x, log_w=None):
    """
    TruFor-style weighted statistics pooling.
    Computes weighted min, max, mean, and mean-square statistics.
    
    Args:
        x: [B, C, H, W] tensor
        log_w: [B, 1, H, W] log weights (optional)
        
    Returns:
        [B, 4*C] tensor containing weighted statistics
    """
    b = x.shape[0]
    c = x.shape[1]
    x = x.view(b, c, -1)
    
    if log_w is None:
        log_w = torch.zeros((b, 1, x.shape[-1]), device=x.device)
    else:
        assert log_w.shape[0] == b
        assert log_w.shape[1] == 1
        log_w = log_w.view(b, 1, -1)
        assert log_w.shape[-1] == x.shape[-1]
    
    log_w = F.log_softmax(log_w, dim=-1)
    x_min = -torch.logsumexp(log_w - x, dim=-1)
    x_max = torch.logsumexp(log_w + x, dim=-1)
    
    w = torch.exp(log_w)
    x_avg = torch.sum(w * x, dim=-1)
    x_msq = torch.sum(w * x * x, dim=-1)
    
    x = torch.cat((x_min, x_max, x_avg, x_msq), dim=1)
    
    return x


class CLipClassifierV10Fusion(nn.Module):
    """
    V10 Fusion: CLIP backbone + LaRE map + handcrafted cues (freq/noise/edges).
    Adds cross-attention fusion, contrastive projection, and an upsampled localization head.
    """
    def __init__(self, num_class=2, clip_type="RN50x64", handcrafted_size: int = 28):
        super(CLipClassifierV10Fusion, self).__init__()
        self.clip_model, _ = clip.load(type_to_path[clip_type], device='cpu', jit=False)

        # Monkey patch to expose block3 features
        self.clip_model.visual.forward = types.MethodType(forward, self.clip_model.visual)
        self.clip_model.encode_image = types.MethodType(encode_image, self.clip_model)

        # Branches
        self.block_proj = nn.Conv2d(2048, 512, kernel_size=1)
        self.lare_proj = nn.Conv2d(4, 256, kernel_size=1)
        self.lare_to_block = nn.Conv2d(256, 512, kernel_size=1)

        self.handcrafted_extractor = MultiFeatureExtractor(target_size=handcrafted_size)
        self.handcrafted_proj = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        # Cross-attention between CLIP tokens (queries) and LaRE tokens (keys/values)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Fusion of (block_proj, attn_out, handcrafted)
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 512 + 128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Heads
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(1024 + 512 + 128, num_class)
        self.proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.loc_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28 -> 56
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 56 -> 112
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.block_proj, self.lare_proj, self.lare_to_block, self.handcrafted_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)
        for m in self.proj_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_input, loss_map, return_loc=False, return_features=False):
        # CLIP backbone with block3 features
        # If CLIP params are trainable and we're in training mode, allow gradients.
        clip_trainable = self.training and any(p.requires_grad for p in self.clip_model.visual.parameters())
        if clip_trainable:
            image_feats, block3_feats = self.clip_model.visual(image_input)
        else:
            with torch.no_grad():
                self.clip_model.eval()
                image_feats, block3_feats = self.clip_model.visual(image_input)

        if block3_feats is None:
            b = image_input.size(0)
            block3_feats = torch.zeros(b, 2048, 28, 28, device=image_input.device, dtype=image_input.dtype)

        # Prepare branches
        block_proj = self.block_proj(block3_feats)

        # Keep LaRE branch dtype aligned under AMP
        lare_28 = F.adaptive_avg_pool2d(loss_map, (28, 28)).to(dtype=block_proj.dtype)
        lare_embed = self.lare_proj(lare_28)
        lare_tokens = self.lare_to_block(lare_embed)

        # Handcrafted branch runs in FP32 (fft stability under AMP)
        handcrafted = self.handcrafted_extractor(image_input)
        # Cast back to AMP dtype for fusion (torch.cat / attention require same dtype)
        handcrafted = handcrafted.to(dtype=block_proj.dtype)
        handcrafted = self.handcrafted_proj(handcrafted)

        # Cross-attention: queries from CLIP block, keys/values from LaRE-guided tokens
        b, c, h, w = block_proj.shape
        queries = block_proj.flatten(2).transpose(1, 2)  # [B, HW, 512]
        kv = lare_tokens.flatten(2).transpose(1, 2)      # [B, HW, 512]
        kv = kv.to(dtype=queries.dtype)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attn_map = attn_out.transpose(1, 2).reshape(b, c, h, w)

        # Fuse and pool
        fused_map = torch.cat([block_proj, attn_map, handcrafted], dim=1)
        fused_map = self.fusion(fused_map)
        global_fused = self.global_pool(fused_map).flatten(1)
        handcrafted_global = self.global_pool(handcrafted).flatten(1)

        logits = self.cls_head(torch.cat([image_feats, global_fused, handcrafted_global], dim=1))

        outputs = [logits]

        if return_loc:
            loc_map = self.loc_head(fused_map)
            outputs.append(loc_map)

        if return_features:
            proj = self.proj_head(global_fused)
            proj = F.normalize(proj, dim=1)
            outputs.append(proj)

        if len(outputs) == 1:
            return outputs[0]
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        return outputs[0], outputs[1], outputs[2]
