import torch
import torch.nn as nn
import math
import copy
from networkTool import device
import torch.nn.functional as F
from networkTool import *

class SelfMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0.):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  # 256//4=64
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  #
        self.qkv = nn.Conv1d(emsize, emsize * 3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.qkv_dwconv = nn.Conv1d(in_channels=emsize * 3, out_channels=emsize * 3, kernel_size=3, stride=1, padding=0,
                                    dilation=0, groups=emsize * 3, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(nhead, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.nhead, 1, self.head_size), mean=0, std=0.02))
        self.seq_length_scale = torch.log(torch.as_tensor(bptt, device='cuda:0'))

    # Slice the output of mlpKQV to implement multi-head attention.
    def slice(self, x):
        new_x_shape = x.size()[:-1] + (self.nhead,self.head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    # em.shape = [bptt,batch_size,emsize]  mask.shape=[bptt, bptt]
    def forward(self, em, mask):
        qkv = self.qkv_dwconv(self.qkv(em.permute(1, 2, 0)))
        k, q, v = qkv.chunk(3, dim=1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        Query = self.slice(q)
        Key = self.slice(k)
        Value = self.slice(v)

        attention_score = ((F.normalize(Query, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(Key, dim=-1).transpose(-2, -1)
        attention_score = attention_score + mask

        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))
        context = torch.matmul(attention_map,Value)

        context = context.permute(0, 2, 1, 3).contiguous()
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0, 1)

        return context

class SG_CM(nn.Module):
    def __init__(self, ninp=256):
        super().__init__()
        self.fc1 = nn.Conv1d(ninp, ninp*8, 1, 1, 0, bias=False)
        self.dwconv = nn.Conv1d(in_channels=ninp*4, out_channels=ninp*4, kernel_size=3, stride=1, padding=0, dilation=0, groups=ninp*4, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv1d(ninp*4, ninp, 1, 1, 0, bias=False)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        input = x
        x, v = self.fc1(x).chunk(2, dim=-2)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        x = input + x
        x = x.permute(2, 0, 1)
        return x

class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.MSA = SelfMultiheadAttention(emsize=ninp,nhead=nhead)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(ninp, eps=1e-5) # It will affect parallel coding
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.SG_CM = SG_CM(ninp)

    # src is the integration of leaf node and its ancestors.
    def forward(self, src, src_mask):

        src2 = self.MSA(src,src_mask)  #Multi-head Attention
        src = self.dropout1(self.norm1(src2)) + src
        src2 = self.dropout(self.SG_CM(src))
        src = src + self.dropout2(self.norm2(src2))
        src = self.norm2(src)
        return src

class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])

    def forward(self,src,src_mask):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output