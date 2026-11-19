#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// A threadgroup of HeadDim threads normalizes one head of one row of the
// packed projection: each query head's queries (and, Gated, its output gate
// after them), then the K and V heads. q_norm and k_norm are read in their
// stored type W: bfloat for an MLX target's, float for a GGUF's F32 norms.
// The pairs of dimensions (i, i + RotaryPairs) then rotate by the row's RoPE
// table row of RotaryPairs pairs, the half split of the first 2 x RotaryPairs
// dimensions: Qwen3.5's first 64, or with HeadDim / 2 pairs the whole head
// (NeoX's). With no pairs (NoPE) the head is stored as normalized, and no
// table is read. With ValueNorm the values are scaled by their inverse RMS
// (Gemma4's v_norm, which has no scale); with SharedKV the packed row holds
// no values and the values are the keys, rotated as stored and scaled by
// their inverse RMS (Gemma4's global layers, whose K serves as V; RoPE is
// orthogonal, so the rotated key's RMS is the key's).
template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs, bool Gated, class W,
          bool ValueNorm = false, bool SharedKV = false>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *chunk_keys, device bfloat *chunk_values,
    FullPrefillParams params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  static_assert(2 * RotaryPairs <= HeadDim, "the rotated pairs are dimensions of the head");
  constexpr uint QStride = Gated ? 2 * HeadDim : HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  constexpr uint PackedStride = QWidth + (SharedKV ? KWidth : 2 * KWidth);
  uint query_tasks = params.tokens * QHeads;
  bool query = task < query_tasks;
  uint local_task = query ? task : task - query_tasks;
  uint heads = query ? QHeads : KHeads;
  uint row = local_task / heads;
  uint head_index = local_task % heads;
  uint position = row;
  device const bfloat *source =
      query ? qkv + ulong(row) * PackedStride + head_index * QStride
            : qkv + ulong(row) * PackedStride + QWidth + head_index * HeadDim;
  device const W *weight = query ? q_norm : k_norm;
  uint kv_head = head_index / (QHeads / KHeads);
  uint local_head = head_index % (QHeads / KHeads);
  device bfloat *destination =
      queries + ((ulong(kv_head) * params.stride + row) *
                     (QHeads / KHeads) +
                 local_head) *
                    HeadDim;
  // Where this head's value dimension thread_index stores its value.
  ulong value_offset = 0;
  if (!query) {
    ulong key_offset =
        (ulong(head_index) * params.stride + position) * HeadDim;
    destination = chunk_keys + key_offset;
    value_offset =
        (ulong(head_index) * HeadDim + thread_index) * params.stride + position;
  }

  float element = float(source[thread_index]);
  const float inverse = rms_inverse_of_sums<HeadDim / 32>(element * element, HeadDim,
                                                          reductions, thread_index, lane,
                                                          simd_group);
  normalized[thread_index] =
      bfloat(element * inverse * float(weight[thread_index]));
  // The inverse RMS of the values this thread stores: their own for a
  // value-normed layer, the normalized key's for a shared-KV one (whose
  // value is the rotated key, of the same RMS).
  float value_inverse = 1.0f;
  if (!query && ValueNorm) {
    const float value = SharedKV ? float(normalized[thread_index])
                                 : float(source[KWidth + thread_index]);
    value_inverse = rms_inverse_of_sums<HeadDim / 32>(value * value, HeadDim, reductions,
                                                       thread_index, lane, simd_group);
    if (!SharedKV)
      chunk_values[value_offset] = bfloat(value * value_inverse);
  } else if (!query) {
    chunk_values[value_offset] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < RotaryPairs) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + RotaryPairs]);
    float cosine = rope_cos[ulong(row) * RotaryPairs + thread_index];
    float sine = rope_sin[ulong(row) * RotaryPairs + thread_index];
    destination[thread_index] = bfloat(first * cosine - second * sine);
    destination[thread_index + RotaryPairs] = bfloat(second * cosine + first * sine);
    if (!query && SharedKV) {
      chunk_values[value_offset] =
          bfloat(float(destination[thread_index]) * value_inverse);
      chunk_values[value_offset + ulong(RotaryPairs) * params.stride] =
          bfloat(float(destination[thread_index + RotaryPairs]) * value_inverse);
    }
  } else if (thread_index >= 2 * RotaryPairs) {
    destination[thread_index] = normalized[thread_index];
    if (!query && SharedKV)
      chunk_values[value_offset] =
          bfloat(float(normalized[thread_index]) * value_inverse);
  }
}
