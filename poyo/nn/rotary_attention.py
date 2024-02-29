import logging
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError:
    xops = None

from poyo.nn.rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool =True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value
        self.using_memory_efficient_attn = use_memory_efficient_attn

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        rotary_time_emb_query,
        rotary_time_emb_context,
        *,
        context_mask=None,
        query_seqlen=None,
        context_seqlen=None,
    ):

        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        if self.using_memory_efficient_attn:
            if context_mask is not None:
                raise NotImplementedError(
                    f"Got non-None `attn_mask`. "
                    f"This implementation with memory efficient attention only works "
                    f"with `x_seqlen` for handling unequal sample lengths. Traditional "
                    f"padding approach is supported with normal non-memory efficient "
                    f"attention."
                )

            if query_seqlen is None or context_seqlen is None:
                raise ValueError(
                    f"Both `query_seqlen` and `context_seqlen` must be valid "
                    f"sequence lengths."
                )

            out = rotary_memory_efficient_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=query_seqlen,
                kv_seqlen=context_seqlen,
            )

        else:
            if query_seqlen is not None or context_seqlen is not None:
                raise NotImplementedError(
                    f"Got non-None `*_seqlen`. "
                    f"You are using torch's attention implementation, which only "
                    f"accepts `attn_mask`."
                    f"If you wish to use seqlen, please use memory efficient "
                    f"attention. "
                )

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=context_mask,
            )
        
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool = True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.using_memory_efficient_attn = use_memory_efficient_attn
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self, 
        x, 
        rotary_time_emb, 
        *,
        x_mask=None,
        x_seqlen=None,
    ):

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        if self.using_memory_efficient_attn:
            if x_mask is not None:
                raise NotImplementedError(
                    f"Got non-None `attn_mask`. "
                    f"This implementation with memory efficient attention only works "
                    f"with `x_seqlen` for handling unequal sample lengths. Traditional "
                    f"padding approach is supported with normal non-memory efficient "
                    f"attention."
                )

            if x_seqlen is None:
                raise ValueError(
                    f"`x_seqlen` must be a valid sequence length."
                )

            out = rotary_memory_efficient_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb,
                rotary_time_emb_kv=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=x_seqlen,
                kv_seqlen=None, # self-attention has the same seqlen for q, k, v
            )

        else:
            if x_seqlen is not None:
                raise NotImplementedError(
                    f"Got non-None `x_seqlen`. "
                    f"You are using torch's attention implementation, which only "
                    f"accepts `attn_mask`."
                    f"If you wish to use `x_seqlen`, please use memory efficient "
                    f"attention. "
                )

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb,
                rotary_time_emb_kv=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=x_mask,
            )
        
        out = self.to_out(out)
        return out


def rotary_default_attention(
    *,
    q, # (b, n_q, (h d), )
    k, # (b, n_kv, (h d), )
    v, # (b, n_kv, (h d), )
    rotary_time_emb_q, # (b, n_q, d)
    rotary_time_emb_kv, # (b, n_kv, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    kv_mask=None, # (b, n_kv)
): # Output: (b, n, (h d), )
    r"""Wraps the default attention implementation with rotary embedding application.
    """

    # default attention expects shape b h n d
    q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)
    v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    # apply rotary embeddings
    q = apply_rotary_pos_emb(rotary_time_emb_q, q, dim=1)
    k = apply_rotary_pos_emb(rotary_time_emb_kv, k, dim=1)
    if rotate_value:
        v = apply_rotary_pos_emb(rotary_time_emb_kv, v, dim=1)

    # attention mask
    if kv_mask is not None:
        kv_mask = rearrange(kv_mask, "b n -> b () () n") 

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=kv_mask, dropout_p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q, out, dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    return out


def rotary_memory_efficient_attention(
    *,
    q, # (n, (h d), )
    k, # (n, (h d), )
    v, # (n, (h d), )
    rotary_time_emb_q, # (n, d)
    rotary_time_emb_kv, # (n, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    q_seqlen=None,
    kv_seqlen=None,
): # Output: (n, (h d), )
    r"""Wraps the memory efficient attention implementation with rotary embedding 
    application.
    """

    # xformers attention expects shape (1, n, h, d, ) 
    q = rearrange(q, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    k = rearrange(k, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    v = rearrange(v, "n (h d) -> n h d", h=num_heads).unsqueeze(0)

    q = apply_rotary_pos_emb(rotary_time_emb_q.unsqueeze(0), q)
    k = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), k)
    if rotate_value:
        v = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), v)

    # Fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        # xformers expects 'list' of seqlens
        if q_seqlen is None:
            raise ValueError(
                f"`q_seqlen` must be a valid sequence length."
            )
        elif isinstance(q_seqlen, torch.Tensor):
            q_seqlen = q_seqlen.tolist()
        elif not isinstance(q_seqlen, list):
            raise ValueError(
                f"`q_seqlen` must be a list or a torch.Tensor, "
                f"got {type(q_seqlen)}"
            )

        if kv_seqlen is not None:
            # xformers expects 'list' of seqlens
            if isinstance(kv_seqlen, torch.Tensor):
                kv_seqlen = kv_seqlen.tolist()
            elif not isinstance(kv_seqlen, list):
                raise ValueError(
                    f"`kv_seqlen` must be a list or a torch.Tensor, "
                    f"got {type(kv_seqlen)}"
                )
            
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    # perform attention, by default will use the optimal attention implementation
    out = xops.memory_efficient_attention(
        q, k, v, attn_bias=attn_bias, p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q.unsqueeze(0), out)

    # return (n, (h d), ), b = 1 is removed
    out = rearrange(out, "b n h d -> b n (h d)").squeeze(0)
    return out