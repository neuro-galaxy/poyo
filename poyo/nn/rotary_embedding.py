import torch
import torch.nn as nn
from einops import repeat, rearrange


class RotaryEmbedding(nn.Module):
    r"""Custom rotary positional embedding layer. This function generates sinusoids of 
    different frequencies, which are then used to modulate the input data. Half of the 
    dimensions are not rotated.

    The frequencies are computed as follows:
    
    .. math::
        f(i) = \frac{2\pi}{t_{\min}} \cdot \frac{t_{\max}}{t_\{min}}^{2i/dim}}

    To rotate the input data, use :func:`apply_rotary_pos_emb`.

    Args:
        dim (int): Dimensionality of the input data.
        t_min (float, optional): Minimum period of the sinusoids.
        t_max (float, optional): Maximum period of the sinusoids.
    """
    def __init__(self, dim, t_min=1e-4, t_max=4.0):
        super().__init__()
        inv_freq = torch.zeros(dim // 2)
        inv_freq[: dim // 4] = (
            2
            * torch.pi
            / (
                t_min
                * (
                    (t_max / t_min)
                    ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))
                )
            )
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        r"""Computes the rotation matrices for given timestamps.
        
        Args:
            timestamps (torch.Tensor): timestamps tensor.
        """
        freqs = torch.einsum("..., f -> ... f", timestamps, self.inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(freqs, x, dim=2):
    r"""Apply the rotary positional embedding to the input data.
    
    Args:
        freqs (torch.Tensor): Frequencies of the sinusoids.
        x (torch.Tensor): Input data.
        dim (int, optional): Dimension along which to rotate.
    """
    dtype = x.dtype
    if dim == 1:
        freqs = rearrange(freqs, "n ... -> n () ...")
    elif dim == 2:
        freqs = rearrange(freqs, "n m ... -> n m () ...")
    x = (x * freqs.cos().to(dtype)) + (rotate_half(x) * freqs.sin().to(dtype))
    return x
