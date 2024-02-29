from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from poyo.nn import RotaryEmbedding, RotaryCrossAttention, RotarySelfAttention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverRotary(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=None,
        dim_head=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        use_memory_efficient_attn=True,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            use_memory_efficient_attn=use_memory_efficient_attn,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
            use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim
        self.using_memory_efficient_attn = self.enc_atn.using_memory_efficient_attn

    def forward(
        self,
        *,      # (   padded   ) or (   chained   )
        inputs, # (B, N_in, dim) or (N_all_in, dim)
        latents, # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries, # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps, # (B, N_in) or (N_all_in,)
        latent_timestamps, # (B, N_latent) or (N_all_latent,)
        output_query_timestamps, # (B, N_out) or (N_all_out,)
        input_mask=None, # (B, N_in) or None
        input_seqlen=None, # None or (B,)
        latent_seqlen=None, # None or (B,)
        output_query_seqlen=None, # None or (B,)
    ) -> Union[
        TensorType["batch", "*nqueries", "dim"], # if padded
        TensorType["ntotal_queries", "dim"], # if chained
    ]:

        # Make sure the arguments make sense
        padded_input = input_mask is not None
        chained_input = (
            input_seqlen is not None
            or latent_seqlen is not None
            or output_query_seqlen is not None
        )

        if padded_input and chained_input:
            raise ValueError(
                f"Cannot specify both input_mask and "
                f"input_seqlen/latent_seqlen/output_query_seqlen."
            )

        if chained_input:
            if (
                input_seqlen is None
                or latent_seqlen is None
                or output_query_seqlen is None
            ):
                raise ValueError(
                    f"Must specify all of input_seqlen, latent_seqlen, "
                    f"output_query_seqlen."
                )

        if padded_input:
            assert inputs.dim() == 3
            assert latents.dim() == 3
            assert output_queries.dim() == 3
            assert input_timestamps.dim() == 2
            assert latent_timestamps.dim() == 2
            assert output_query_timestamps.dim() == 2
            assert input_mask.dim() == 2

        if chained_input:
            assert inputs.dim() == 2
            assert latents.dim() == 2
            assert output_queries.dim() == 2
            assert input_timestamps.dim() == 1
            assert latent_timestamps.dim() == 1
            assert output_query_timestamps.dim() == 1
            assert input_seqlen.dim() == 1
            assert latent_seqlen.dim() == 1
            assert output_query_seqlen.dim() == 1

        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            context_mask=input_mask, # used if default attention
            query_seqlen=latent_seqlen, # used if memory efficient attention
            context_seqlen=input_seqlen, # used if memory efficient attention
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(latents, latent_timestamp_emb, x_seqlen=latent_seqlen)
            )
            latents = latents + self.dropout(self_ff(latents))

        if output_queries is None:
            return latents

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries, 
            latents, 
            output_timestamp_emb, 
            latent_timestamp_emb,
            context_mask=None,
            query_seqlen=output_query_seqlen,
            context_seqlen=latent_seqlen,
        )
        output_queries = output_queries + self.dec_ffn(output_queries)

        return output_queries
