import numpy as np
import torch.nn as nn
import torch
from modeling.sem_seg_head.daformer_head import resize


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=[64, 128, 320, 512], embedding_dim=768, num_classes=19, in_keys=['s3'], dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()

        assert len(in_channels) == len(in_keys)
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.in_keys = in_keys

        self.linear_list = nn.ModuleList()
        for in_c in self.in_channels:
            self.linear_list.append(MLP(input_dim=in_c, embed_dim=embedding_dim))

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(len(in_channels) * embedding_dim, embedding_dim, kernel_size=1),
            nn.GroupNorm(num_channels=embedding_dim, num_groups=32),
            nn.SiLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def transfer_input_dict_to_list(self, inputs_dict):
        inputs_list = []
        for key in self.in_keys:
            inputs_list.append(inputs_dict[key])
        # ensure all features from backbone are used
        assert len(inputs_list) == len(inputs_dict)
        return inputs_list

    def forward(self, inputs):

        input_features = inputs['output_features']
        x_list = self.transfer_input_dict_to_list(input_features)
        n, _, h, w = x_list[-1].shape

        feats = []
        for i in range(len(x_list)):
            feat = self.linear_list[i](x_list[i]).permute(0,2,1).reshape(n, -1, x_list[i].shape[2], x_list[i].shape[3])
            feat = resize(feat, size=(h, w), mode='bilinear',align_corners=False)
            feats.append(feat)
        x = torch.cat(feats, dim=1)

        x = self.linear_fuse(x)

        x = self.dropout(x)
        x = self.linear_pred(x)

        return x