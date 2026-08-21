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

def softmax(in_features,dim):
    max_feature,_ = torch.max(in_features,dim=dim,keepdim=True)
    nor_features = in_features - max_feature
    e_features = torch.exp(nor_features)
    sum_features = torch.sum(e_features,dim=dim,keepdim=True)
    out_features = e_features/sum_features
    return out_features

def scaled_dot_product_attention(Q,K,V,mask=None):
    logits = torch.einsum("...ij,...kj->...ik",Q,K) * (Q.shape[-1]**-0.5)
    if mask is None:
            ones = torch.ones_like(logits)
            tril = torch.tril(ones)
            mask = tril==1
    logits = logits.masked_fill(~mask,float("-inf"))
    pro = softmax(logits,dim=-1)
    atten = torch.einsum("...ij,...jk -> ...ik",pro,V)
    return atten

def multihead_self_attention(d_model: int,num_heads: int,q_proj_weight: float,k_proj_weight: float,v_proj_weight: float,o_proj_weight: float,in_features: float) -> float:
    Q = torch.einsum("...ij,kj -> ...ik",in_features,q_proj_weight)
    K = torch.einsum("...ij,kj -> ...ik",in_features,k_proj_weight)
    V = torch.einsum("...ij,kj -> ...ik",in_features,v_proj_weight)

    assert d_model%num_heads==0
    head_size = d_model//num_heads
    shape = Q.shape[:-1]
    split_Q = Q.view(*shape,num_heads,head_size).transpose(-3,-2)
    split_K = K.view(*shape,num_heads,head_size).transpose(-3,-2)
    split_V = V.view(*shape,num_heads,head_size).transpose(-3,-2)

    atten = scaled_dot_product_attention(split_Q,split_K,split_V).transpose(-3,-2).reshape(*shape,-1)
    out_features = torch.einsum("...ij,kj -> ...ik",atten,o_proj_weight)

    return out_features

def multihead_self_attention_with_rope(d_model: int,num_heads: int,max_seq_len:int,theta:float,q_proj_weight: float,k_proj_weight: float,v_proj_weight: float,o_proj_weight: float,in_features: float,token_positions:int) -> float:
    Q = torch.einsum("...ij,kj -> ...ik",in_features,q_proj_weight)
    K = torch.einsum("...ij,kj -> ...ik",in_features,k_proj_weight)
    V = torch.einsum("...ij,kj -> ...ik",in_features,v_proj_weight)

    assert d_model%num_heads==0
    head_size = d_model//num_heads
    shape = Q.shape[:-1]
    split_Q = Q.view(*shape,num_heads,head_size).transpose(-3,-2)
    split_K = K.view(*shape,num_heads,head_size).transpose(-3,-2)
    split_V = V.view(*shape,num_heads,head_size).transpose(-3,-2)

    rope = RotaryPositionalEmbedding(theta,head_size,max_seq_len)
    out_query = rope(split_Q,token_positions)
    out_key = rope(split_K,token_positions)

    atten = scaled_dot_product_attention(out_query,out_key,split_V).transpose(-3,-2).reshape(*shape,-1)
    out_features = torch.einsum("...ij,kj -> ...ik",atten,o_proj_weight)

    return out_features
