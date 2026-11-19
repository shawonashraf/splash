#include <metal_stdlib>
using namespace metal;

// The final logit softcap, s * tanh(l / s), in place over `count` logits.
kernel void softcap_logits(device float *logits [[buffer(0)]],
                           constant float &s [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
  if (s > 0.0f) logits[index] = s * tanh(logits[index] / s);
}
