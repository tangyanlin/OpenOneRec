# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fallback implementations for flash_attn functions when flash_attn is not installed.

This module provides pure-PyTorch implementations of:
- index_first_axis: Select elements from a tensor using indices along the first axis
- pad_input: Pad unpadded (flat) tensor back to padded (batched) form
- unpad_input: Remove padding from a padded (batched) tensor to get flat valid tokens
- rearrange: Re-exported from einops
- apply_rotary_emb: Apply rotary embeddings (fallback for flash_attn.layers.rotary.apply_rotary_emb)
- flash_attn_varlen_func: Flash attention with variable sequence lengths (raises NotImplementedError)

These are used by the actor/critic training code when use_remove_padding=True,
and by megatron parallel attention layers.
"""

import torch

try:
    from einops import rearrange
except ImportError:
    # Minimal fallback for rearrange if einops is also not available
    def rearrange(tensor, pattern, **kwargs):
        raise ImportError(
            "einops is required for rearrange when flash_attn is not installed. "
            "Please install it: pip install einops"
        )


def index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select elements from tensor using indices along the first axis (batch * seq).

    Equivalent to flash_attn.bert_padding.index_first_axis.

    Args:
        tensor: Tensor of shape (batch * seq, ...) to index into.
        indices: Long tensor of indices to select.

    Returns:
        Indexed tensor of shape (len(indices), ...).
    """
    return tensor[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Remove padding from a padded tensor to get flat valid tokens.

    Equivalent to flash_attn.bert_padding.unpad_input.

    Args:
        hidden_states: Tensor of shape (batch, seq, ...).
        attention_mask: Binary mask of shape (batch, seq), 1 for valid tokens.

    Returns:
        Tuple of (unpadded_tensor, indices, cu_seqlens, max_seqlen_in_batch).
        - unpadded_tensor: shape (total_valid_tokens, ...)
        - indices: shape (total_valid_tokens,), indices into flattened (batch*seq) dimension
        - cu_seqlens: shape (batch+1,), cumulative sequence lengths
        - max_seqlen_in_batch: int, maximum sequence length in the batch
    """
    batch_size, seq_len = attention_mask.shape
    # Get indices of valid tokens in the flattened (batch * seq) space
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).squeeze(-1)

    # Gather the valid tokens
    # hidden_states shape: (batch, seq, ...) -> flatten to (batch*seq, ...) -> index
    flat_hidden = hidden_states.reshape(-1, *hidden_states.shape[2:])
    unpadded_tensor = flat_hidden[indices]

    # Compute cumulative sequence lengths
    seqlens_in_batch = attention_mask.sum(dim=1).long()  # (batch,)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return unpadded_tensor, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Pad an unpadded (flat) tensor back to padded (batched) form.

    Equivalent to flash_attn.bert_padding.pad_input.

    Args:
        hidden_states: Unpadded tensor of shape (total_valid_tokens, ...).
        indices: Indices of valid tokens in the flattened (batch*seq) space.
        batch: Batch size.
        seqlen: Sequence length.

    Returns:
        Padded tensor of shape (batch, seqlen, ...), with zeros in padded positions.
    """
    output = torch.zeros(
        batch * seqlen, *hidden_states.shape[1:],
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Fallback for flash_attn.layers.rotary.apply_rotary_emb.

    This implements the rotary embedding application using pure PyTorch.
    The flash_attn version uses a custom CUDA kernel; this fallback uses
    the standard approach: x * cos + rotate_half(x) * sin.

    Args:
        x: Input tensor of shape (total_seq_len, num_heads, head_dim) or (batch, seq, num_heads, head_dim).
        cos: Cosine of rotary embeddings, shape compatible with x.
        sin: Sine of rotary embeddings, shape compatible with x.
        interleaved: If True, uses interleaved layout (not supported in fallback, ignored).
        inplace: If True, modifies x in-place (not supported in fallback, ignored).
        cu_seqlens: Cumulative sequence lengths for varlen mode (not used in fallback).
        max_seqlen: Maximum sequence length for varlen mode (not used in fallback).

    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    if interleaved:
        # For interleaved mode, we need to handle the layout differently
        # Interleaved means the rotary dims are interleaved as [d0, d1, d0, d1, ...]
        # Non-interleaved means [d0, d0, d1, d1, ...]
        # We de-interleave, apply rotary, then re-interleave
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos_half = cos[..., :cos.shape[-1] // 2] if cos.shape[-1] == x.shape[-1] else cos
        sin_half = sin[..., :sin.shape[-1] // 2] if sin.shape[-1] == x.shape[-1] else sin
        out1 = x1 * cos_half - x2 * sin_half
        out2 = x2 * cos_half + x1 * sin_half
        out = torch.stack((out1, out2), dim=-1).flatten(-2)
    else:
        # Standard non-interleaved rotary embedding
        # rotate_half: split the last dim in half and negate the first half
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        # cos and sin may be half the dim or full dim
        if cos.shape[-1] == x.shape[-1] // 2:
            cos1 = cos
            sin1 = sin
        else:
            cos1 = cos[..., :cos.shape[-1] // 2]
            sin1 = sin[..., :sin.shape[-1] // 2]
        out1 = x1 * cos1 - x2 * sin1
        out2 = x2 * cos1 + x1 * sin1
        out = torch.cat((out1, out2), dim=-1)

    return out


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
    return_softmax=False, block_table=None,
):
    """Fallback for flash_attn.flash_attn_varlen_func.

    This function requires flash_attn's custom CUDA kernel for efficient
    variable-length attention computation. There is no simple pure-PyTorch
    fallback that maintains the same interface and performance.

    Raises:
        NotImplementedError: Always. Install flash_attn to use this function.
    """
    raise NotImplementedError(
        "flash_attn_varlen_func requires flash_attn to be installed. "
        "This is used by megatron parallel attention layers which require "
        "flash_attn for efficient variable-length attention. "
        "Please install flash_attn: pip install flash-attn"
    )
