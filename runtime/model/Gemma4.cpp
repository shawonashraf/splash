#include "model/Gemma4.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {

namespace {

// A layer's attention of its own geometry: the sliding layers' 32x256
// queries over 16x256 keys and values; the global layers' 32x512 queries
// over 4x512 keys, shared as values (no attn_v). The q and k norms are of the
// layer's head width.
QwenMixerWeights readGemma4Attention(WeightFile &file, const BlockTargetFormat &format,
                                         const Gemma4Layout &layout, bool global) {
  QwenAttentionWeights attention;
  const uint32_t headDim = global ? layout.globalAttentionHeadDimension : layout.attentionHeadDimension;
  const uint64_t qRows = uint64_t{layout.attentionQueryHeads} * headDim;
  const uint64_t kvRows = uint64_t{global ? layout.globalAttentionKvHeads : layout.attentionKvHeads} * headDim;
  attention.inputProjection =
      format.fused(file, qRows + (global ? kvRows : 2 * kvRows), layout.hiddenSize,
                   global ? std::initializer_list<std::string_view>{"attn-q", "attn-k"}
                          : std::initializer_list<std::string_view>{"attn-q", "attn-k", "attn-v"});
  attention.queryNorm = format.norm(file, headDim, "query-norm");
  attention.keyNorm = format.norm(file, headDim, "key-norm");
  attention.outputProjection = format.projection(file, layout.hiddenSize, qRows, "attention-output");
  return attention;
}

} // namespace

// A layer image holds its input norm, its attention of its geometry, the norm
// after it, the FFN's norm and dense SwiGLU, the norm after the FFN and its
// output scale (Gemma4Gguf). The head holds the final norm and the rope
// frequency table; the embedding image the token table, which is also the
// logits projection (tied).
Gemma4Weights loadGemma4Weights(metal::MetalBackend &backend, Gemma4Layout layout,
                                const QwenTargetFiles &files) {
  return loadQwenTarget<Gemma4Weights>(
      backend, layout, files, [&](auto &gguf, const BlockTargetFormat &format, const MoeExperts &) {
        Gemma4Weights result;
        result.layout = layout;
        result.layers.reserve(layout.layers);
        for (uint32_t index = 0; index < layout.layers; ++index) {
          WeightFile file = gguf.layer(index);
          Gemma4LayerWeights &layer = result.layers.emplace_back();
          const bool global = !layout.kvLayout().windowed(index);
          layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
          layer.mixer = readGemma4Attention(file, format, layout, global);
          layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
          layer.ffnNorm = format.norm(file, layout.hiddenSize, "ffn-norm");
          layer.gateProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
          layer.upProjection = format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
          layer.downProjection = format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
          layer.postFfnNorm = format.norm(file, layout.hiddenSize, "post-ffw-norm");
          layer.outputScale = file.section(kFloat32Bytes, "layer-output-scale");
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = gguf.head();
          result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = gguf.embedding();
          result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
          // Tied embeddings: the head runs over the table's stored rows,
          // decoded to fp32 as a float segment's.
          result.logitsProjection = {layout.vocabularySize, layout.hiddenSize,
                                     ops::BlockWeights{{ops::QuantizedSegment::rows(
                                         result.tokenEmbedding.rows().formatId, layout.vocabularySize,
                                         layout.hiddenSize, result.tokenEmbedding.rows().rows)}}};
          result.logitsProjection.destination = ops::FloatOutput::Float32;
          file.finish();
          result.files.push_back(file.record());
        }
        return result;
      });
}

const ops::BlockMoeWeights *qwenMoeBlock(const Gemma4LayerWeights &) noexcept { return nullptr; }
std::optional<ops::SwiGluProjections> denseFfn(const Gemma4LayerWeights &layer) noexcept {
  return ops::SwiGluProjections{&layer.gateProjection, &layer.upProjection, &layer.downProjection};
}

std::optional<float> layerOutputScale(const Gemma4LayerWeights &layer) noexcept {
  if (!layer.outputScale) return std::nullopt;
  return std::optional<float>{*static_cast<const float *>(layer.outputScale.contents())};
}

} // namespace splash::model
