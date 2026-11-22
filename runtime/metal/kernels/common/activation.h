#pragma once
#include <metal_stdlib>
using namespace metal;

// SiLU and the logistic sigmoid of every kernel, e^-x taken as
// fast::exp2(-x * log2(e)). A gated product multiplies splash_silu(gate)
// into its other factor, so silu(g) * up evaluates g / (1 + e^-g) * up.
inline float splash_silu(float x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float2 splash_silu(float2 x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float splash_sigmoid(float x) {
  return 1.0f / (1.0f + fast::exp2(-1.44269504089f * x));
}
// Gemma4's FFN gate, GELU with the tanh approximation. The argument is clamped
// to +/-8, past which tanh is 1 to within 3e-7: precise tanh miscomputes large
// arguments under -std=metal4.0 -O3, and fast::tanh saturates to exactly +/-1.
inline float splash_gelu_tanh(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + fast::tanh(clamp(inner, -8.0f, 8.0f)));
}
