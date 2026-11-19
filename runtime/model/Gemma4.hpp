#pragma once

// Gemma4 (unsloth/gemma-4-31B-it-GGUF; GGUF architecture gemma4, model type
// gemma4_text): a pre-norm decoder of plain attention and dense SwiGLU
// layers, 60 of them every one attending (a layout of the Qwen hybrid's
// with no GDN layers, as Flint's is), which differ in two ways from every
// other family:
//  - Two attention geometries. 50 layers slide: 32 query heads over 16 KV
//    heads of 256, a window of the 1024 keys that end at a row, RoPE over
//    whole heads of the sliding head width (theta 1e4). Every 6th layer (5,
//    11, ..., 59) is global: 32 query heads over 4 KV heads of 512, every
//    key, RoPE theta 1e6. The KV pages hold the sliding geometry, 16 heads
//    of 256; a global layer's 4 heads of 512 store as 8 pseudo-heads of 256
//    in the same page, so one kv::Layout serves both.
//  - Each layer's q and k carry their own RMS norm (of the head width), and
//    the logits pass through a softcap, s * tanh(l / s) with
//    final_logit_softcapping s.
// Its weights load from a GGUF only.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace splash::model {

// Gemma4's sliding layers, a bit per layer: every layer but every 6th, which
// is global (full attention).
inline constexpr uint64_t kGemma4SlidingLayers = [] {
  uint64_t layers = 0;
  for (uint32_t layer = 0; layer < 60; ++layer)
    if (layer % 6 != 5) layers |= uint64_t{1} << layer;
  return layers;
}();

struct Gemma4Layout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture; no safetensors
  // checkpoint of it loads.
  static constexpr std::string_view modelType = "gemma4_text";
  static constexpr std::string_view ggufArchitecture = "gemma4";
  static constexpr bool loadsSafetensors = false;

  // The global layers' KV heads and head width (the sliding layers' are the
  // base's attentionKvHeads and attentionHeadDimension), their RoPE base.
  uint32_t globalAttentionKvHeads = 4;
  uint32_t globalAttentionHeadDimension = 512;
  float globalRotaryTheta = 1'000'000.0F;

  // Gemma4-31B's, its first family (kFamilies).
  constexpr Gemma4Layout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 1'010'000,
                          .layers = 60,
                          .hiddenSize = 5376,
                          .vocabularySize = 262'144,
                          .attentionWidth = 8192,
                          .attentionQueryHeads = 32,
                          .attentionKvHeads = 16,
                          .attentionHeadDimension = 256,
                          .attentionGate = false,
                          .rotaryPairs = 128,
                          .rotaryTheta = 10'000.0F,
                          .slidingWindow = 1024,
                          .slidingLayers = kGemma4SlidingLayers,
                          .fullAttentionPeriod = 1,
                          .dualAttentionGeometry = true,
                          .valueNormed = true,
                          .logitSoftcapping = 30.0F,
                          .maskToken = 4,
                          .stopTokens = {106},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 21'504},
                         "Gemma4"} {}
  bool operator==(const Gemma4Layout &) const = default;
};



// A layer: its input norm, its attention (the q/k/v projections of its own
// geometry, the q and k norms of its head width, its output projection), the
// norm after it, the FFN's norm, its dense SwiGLU, the norm after the FFN and
// its learned output scale (a float per layer).
struct Gemma4LayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::NormWeights ffnNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
  ops::NormWeights postFfnNorm;
  metal::MetalBuffer outputScale;
};

using Gemma4Weights = QwenTargetWeights<Gemma4Layout, Gemma4LayerWeights>;

[[nodiscard]] Gemma4Weights loadGemma4Weights(metal::MetalBackend &backend, Gemma4Layout layout,
                                              const QwenTargetFiles &files);

// Its every layer's FFN is dense.
[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const Gemma4LayerWeights &) noexcept;
[[nodiscard]] std::optional<ops::SwiGluProjections> denseFfn(const Gemma4LayerWeights &layer) noexcept;
// The layer's learned output scale (HF layer_scalar), which multiplies the
// whole layer output after its residuals.
[[nodiscard]] std::optional<float> layerOutputScale(const Gemma4LayerWeights &layer) noexcept;

} // namespace splash::model
