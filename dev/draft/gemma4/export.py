#!/usr/bin/env python3
"""Export a trained DFlash2 draft checkpoint in the layout the engine loads
(runtime/model/DraftCheckpoint.cpp, readDraftLayout in ModelDescriptor.mm):

  config.json                 DFlash2DraftModel / qwen3 with dflash_config
  model.safetensors           BF16 tensors (single file)
  mask_embedding.safetensors  one BF16 tensor `mask_embedding` of the hidden size

Usage:
  .venv/bin/python dev/draft/gemma4/export.py \
      --checkpoint /tmp/g4-draft/checkpoint-final.pt --out /tmp/g4-model/draft \
      --target-layers 11 29 47
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

H = 5376
HEAD_DIM = 128
Q_HEADS = 32
KV_HEADS = 8
ATTN_W = Q_HEADS * HEAD_DIM
KV_W = KV_HEADS * HEAD_DIM
INTER = 6144
TAPS = 2
STAGES = 2
GROUP = 16
GROUPS = H // GROUP
DYN = STAGES * TAPS * GROUPS
RANK = 256
TOPK = 16
VOCAB = 262144
MASK_TOKEN = 4
BLOCK = 8
WINDOW = 2048


def bf16(t):
    return t.detach().to(torch.bfloat16).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-layers", type=int, nargs="+", required=True)
    ap.add_argument("--intermediate", type=int, default=INTER)
    ap.add_argument("--rope-theta", type=float, default=10_000.0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state"]
    k = len(args.target_layers)
    assert state["fc.weight"].shape[1] == k * H, state["fc.weight"].shape

    tensors = {
        "fc.weight": bf16(state["fc.weight"]),
        "hidden_norm.weight": bf16(state["hidden_norm.weight"]),
        "norm.weight": bf16(state["norm.weight"]),
        "layers.0.input_layernorm.weight": bf16(state["layer.input_layernorm.weight"]),
        "layers.0.attention_conv.base_kernel": bf16(state["layer.attention_conv.base_kernel"]),
        "layers.0.attention_conv.kernel_projection.weight": bf16(state["layer.attention_conv.kernel_projection.weight"]),
        "layers.0.self_attn.q_proj.weight": bf16(state["layer.q_proj.weight"]),
        "layers.0.self_attn.k_proj.weight": bf16(state["layer.k_proj.weight"]),
        "layers.0.self_attn.v_proj.weight": bf16(state["layer.v_proj.weight"]),
        "layers.0.self_attn.o_proj.weight": bf16(state["layer.o_proj.weight"]),
        "layers.0.self_attn.q_norm.weight": bf16(state["layer.q_norm.weight"]),
        "layers.0.self_attn.k_norm.weight": bf16(state["layer.k_norm.weight"]),
        "layers.0.post_attention_layernorm.weight": bf16(state["layer.post_attention_layernorm.weight"]),
        "layers.0.mlp_conv.base_kernel": bf16(state["layer.mlp_conv.base_kernel"]),
        "layers.0.mlp_conv.kernel_projection.weight": bf16(state["layer.mlp_conv.kernel_projection.weight"]),
        "layers.0.mlp.gate_proj.weight": bf16(state["layer.gate_proj.weight"]),
        "layers.0.mlp.up_proj.weight": bf16(state["layer.up_proj.weight"]),
        "layers.0.mlp.down_proj.weight": bf16(state["layer.down_proj.weight"]),
        "candidate_selector.hidden_projection.weight": bf16(state["selector_hidden.weight"]),
        "candidate_selector.predecessor_codebook": bf16(state["predecessor_codebook"]),
        "candidate_selector.successor_codebook": bf16(state["successor_codebook"]),
    }
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": H,
        "vocab_size": VOCAB,
        "num_hidden_layers": 1,
        "intermediate_size": args.intermediate,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "sliding_window": WINDOW,
        "rms_norm_eps": 1e-6,
        "is_causal": False,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": args.rope_theta},
        "dflash_config": {
            "block_size": BLOCK,
            "conv_group_size": GROUP,
            "conv_kernel_size": TAPS,
            "selector_rank": RANK,
            "selector_top_k": TOPK,
            "mask_token_id": MASK_TOKEN,
            "target_layer_ids": args.target_layers,
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    save_file({"mask_embedding": bf16(state["mask_embedding"])},
              str(out / "mask_embedding.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"exported draft to {out}")


if __name__ == "__main__":
    main()
