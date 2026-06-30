import torch
import triton
import triton.language as tl

@triton.jit
def _round_half_away_from_zero(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quantize_int8(x, scale):
    q = _round_half_away_from_zero(x / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    return q.to(tl.int8)


# =============================================================================
# N token 1 head opt: [BLOCK_T, dim]
# =============================================================================

@triton.jit
def _fused_indexer_q_rope_int8_quant_ntoken_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0: tl.constexpr,
    index_q_stride1: tl.constexpr,
    cos_sin_ptr,
    cos_sin_stride: tl.constexpr,
    index_q_int8_ptr,
    index_q_int8_stride0: tl.constexpr,
    index_q_int8_stride1: tl.constexpr,
    index_weights_ptr,
    index_weights_stride: tl.constexpr,
    index_weights_out_ptr,
    index_weights_out_stride: tl.constexpr,
    weights_scale: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_block = tl.program_id(0)
    head_idx = tl.program_id(1)

    HALF_ROT_DIM: tl.constexpr = 32
    NOPE_DIM: tl.constexpr = 64

    tok_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = tok_offsets < NUM_TOKENS

    half_offset = tl.arange(0, HALF_ROT_DIM)
    nope_offset = tl.arange(0, NOPE_DIM)

    pos = tl.load(pos_ptr + tok_offsets, mask=tok_mask, other=0)

    cos = tl.load(
        cos_sin_ptr + pos[:, None] * cos_sin_stride + half_offset[None, :],
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    sin = tl.load(
        cos_sin_ptr
        + pos[:, None] * cos_sin_stride
        + half_offset[None, :]
        + HALF_ROT_DIM,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    q_base = (
        index_q_ptr
        + tok_offsets[:, None] * index_q_stride0
        + head_idx * index_q_stride1
    )

    x_nope = tl.load(
        q_base + nope_offset[None, :],
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    rot_base = q_base + NOPE_DIM

    x_even = tl.load(
        rot_base + half_offset[None, :] * 2,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    x_odd = tl.load(
        rot_base + half_offset[None, :] * 2 + 1,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax_nope = tl.max(tl.abs(x_nope), axis=1)
    amax_even = tl.max(tl.abs(r_even), axis=1)
    amax_odd = tl.max(tl.abs(r_odd), axis=1)
    amax = tl.maximum(amax_nope, tl.maximum(amax_even, amax_odd))

    q_scale = tl.where(amax > 0.0, amax / 127.0, 1.0)

    out_base = (
        index_q_int8_ptr
        + tok_offsets[:, None] * index_q_int8_stride0
        + head_idx * index_q_int8_stride1
    )

    tl.store(
        out_base + nope_offset[None, :],
        _quantize_int8(x_nope, q_scale[:, None]),
        mask=tok_mask[:, None],
    )

    out_rot_base = out_base + NOPE_DIM

    tl.store(
        out_rot_base + half_offset[None, :] * 2,
        _quantize_int8(r_even, q_scale[:, None]),
        mask=tok_mask[:, None],
    )

    tl.store(
        out_rot_base + half_offset[None, :] * 2 + 1,
        _quantize_int8(r_odd, q_scale[:, None]),
        mask=tok_mask[:, None],
    )

    weight = tl.load(
        index_weights_ptr + tok_offsets * index_weights_stride + head_idx,
        mask=tok_mask,
        other=0.0,
    ).to(tl.float32)

    weight_out = weight * q_scale * weights_scale

    tl.store(
        index_weights_out_ptr + tok_offsets * index_weights_out_stride + head_idx,
        weight_out,
        mask=tok_mask,
    )


def fused_indexer_q_rope_int8_quant(
    positions,
    index_q,
    cos_sin_cache,
    index_weights,
    softmax_scale,
    head_scale,
    num_warps: int,
):
    positions = positions.contiguous()
    index_q = index_q.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()
    index_weights = index_weights.contiguous()

    T, H, D = index_q.shape
    assert H == 64
    assert D == 128
    assert cos_sin_cache.shape[-1] == 64

    index_q_int8 = torch.empty_like(index_q, dtype=torch.int8)
    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    weights_scale = float(softmax_scale * head_scale)

    kernel = _fused_indexer_q_rope_int8_quant_ntoken_opt_kernel
    if T < 2048:
        grid = (triton.cdiv(T, 4), H)
        kernel[grid](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            index_q_int8,
            index_q_int8.stride(0),
            index_q_int8.stride(1),
            index_weights,
            index_weights.stride(0),
            index_weights_out,
            index_weights_out.stride(0),
            weights_scale,
            T,
            BLOCK_T=4,
            num_warps=1,
        )
    else:
        grid = (triton.cdiv(T, 2), H)
        kernel[grid](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            index_q_int8,
            index_q_int8.stride(0),
            index_q_int8.stride(1),
            index_weights,
            index_weights.stride(0),
            index_weights_out,
            index_weights_out.stride(0),
            weights_scale,
            T,
            BLOCK_T=2,
            num_warps=1,
        )


    return index_q_int8, index_weights_out
