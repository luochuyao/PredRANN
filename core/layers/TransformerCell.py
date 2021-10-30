import torch
import torch.nn as nn
from core.layers.Attention import *
from core.layers.tool import *


class TransformerCell(nn.Module):
    def __init__(self,qin_channels,kvin_channels, heads, head_channels, width):
        super(TransformerCell, self).__init__()
        self.qin_channels = qin_channels
        self.kvin_channels = kvin_channels
        self.heads = heads
        self.head_channels = head_channels
        self.width = width
        self.inner_channels = self.head_channels * self.heads
        self.channel_attn = FullAttention(mask_flag=False, factor=5, attention_dropout=0.2)
        self.k_projection = TimeDistribution(
            model=nn.Conv2d(
                in_channels = self.kvin_channels,
                out_channels = self.inner_channels,
                kernel_size=1,
                padding=0
            )
        )
        self.v_projection = TimeDistribution(
            model=nn.Conv2d(
                in_channels=self.kvin_channels,
                out_channels=self.inner_channels,
                kernel_size=1,
                padding=0
            )
        )
        self.q_projection = nn.Conv2d(
                in_channels = self.qin_channels,
                out_channels = self.inner_channels,
                kernel_size=1,
                padding=0
            )
        self.output_projection = nn.Conv2d(
                in_channels = self.inner_channels,
                out_channels = self.qin_channels,
                kernel_size = 3,
                padding = 1
            )

        self.norm = nn.LayerNorm([qin_channels, width, width])


    def forward(self, in_query, key, value):
        if type(key)==type([]):
            key = torch.stack(key,1)
            value = torch.stack(value, 1)
        query = self.q_projection(in_query)
        key = self.k_projection(key)
        value = self.v_projection(value)

        B,T,_,H,W = key.shape

        query = query.view(B, 1, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))
        key = key.view(B, T, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))
        value = value.view(B, T, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))

        query = query.reshape(B, 1*self.head_channels,self.heads, H*W)
        key = key.reshape(B, T*self.head_channels, self.heads, H*W)
        value = value.reshape(B, T*self.head_channels, self.heads,  H*W)

        s_attn = self.channel_attn(query, key, value,None)
        s_attn = s_attn.view(B, 1, self.head_channels,self.heads, H, W)
        s_attn = s_attn.reshape(B, 1*self.heads*self.head_channels, H, W)

        output = self.output_projection(s_attn)
        output = self.norm(in_query + output)

        return output