import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2/(in_features+out_features)) ** 0.5
        trunc_lower = -3 * std
        trunc_upper = 3 * std
        self.weight = torch.empty((out_features,in_features),device=device,dtype=dtype)
        self.weight = nn.Parameter(self.weight)
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=trunc_lower,b=trunc_upper)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("...j,kj -> ...k",[x,self.weight]) 
        # y = x @ self.weight.T
        return y

class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        std = 1
        trunc_lower = -3
        trunc_upper = 3
        self.weight = torch.empty((vocab_size,d_model))
        self.weight = nn.Parameter(self.weight)
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=trunc_lower,b=trunc_upper)
    def forward(self,token_id):
        return self.weight[token_id]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.ones(d_model,device=device,dtype=dtype)
        self.d_model = d_model
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入shape (B, L, d_model)，输出同shape
        type_ = x.dtype
        x = x.to(torch.float32)
        sum = torch.sum(x**2,keepdim=True,dim=-1)
        mean = sum/self.d_model
        coeff = (mean + self.eps) ** -0.5
        num = self.weight * x
        y = num * coeff
        y = y.to(type_)
        return y

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w1 = Linear(d_model,d_ff)
        self.w2 = Linear(d_ff,d_model)
        self.w3 = Linear(d_model,d_ff)
    def forward(self,x):
        b1 = F.silu(self.w1(x))
        b2 = self.w3(x)
        sum = torch.einsum("...j,...j -> ...j",b1,b2)
        y = self.w2(sum)
        return y

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        assert d_k%2==0,"维度为奇数，要求为偶数"
        super().__init__()
        n_half = d_k//2
        freq = theta**(-torch.arange(n_half,device=device)/n_half)
        seq_pos = torch.arange(max_seq_len,device=device)
        mtheta = torch.einsum("i,j->ij",seq_pos,freq)
        self.register_buffer("mtheta",mtheta)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (*, seq_len, d_k)
        pos_mtheta = self.mtheta[token_positions]
        new_x = x.view(*x.shape[:-1],-1,2)
        x1 = new_x[...,0]
        x2 = new_x[...,1]
        cos = torch.cos(pos_mtheta)
        sin = torch.sin(pos_mtheta)
        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos 
        y = torch.stack([new_x1,new_x2],dim=-1).view(*x.shape[:-1],-1)
        return y

