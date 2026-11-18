# Training a DFlash2 draft for Gemma4

Gemma4-31B has no published DFlash2 draft, so Splash cannot pair one from
its usual repository yet (`incoai/Gemma-4-31B-DFlash2` is the placeholder the
family names). This is the spec for training one ourselves, matching the
draft contract the engine checks at every start (`readDraftLayout`,
`docs/models.md#drafts`).

## Target and capture

- Teacher: `unsloth/gemma-4-31B-it-GGUF` (any UD variant; Q4_K_M for
  capture fidelity), served by Splash itself or llama.cpp — the draft only
  needs the teacher's hidden states and logits, not its weights.
- Capture teacher-forced activations on a coding-agent corpus (the same
  SPEED-Bench prompts the launch benchmarks use, plus multi-turn tool-call
  transcripts in Gemma4's chat format): run the teacher with
  `dev/benchmarks/teacher_logits.mm --captures yes`, which records each
  layer's input/output hidden states and logits per token.
- Hidden width 5376 is 16 × 336, so the draft's convolution groups need no
  new kernels: any hidden width of whole 16-channel groups loads.

## Draft architecture (the kernels' fixed shape)

Exactly the shape `readDraftLayout` holds to the kernels:

| Field | Value |
| --- | --- |
| `model_type` / architectures | `qwen3` / `DFlash2DraftModel` (the draft arch is shared) |
| `hidden_size` | 5376 (the target's) |
| `vocab_size` | 262144 (the target's, tied) |
| `num_hidden_layers` | 1 |
| attention | 32 query heads, 8 KV heads, head dim 128, `sliding_window` 2048, `rope_theta` 10000 |
| FFN | `intermediate_size` 6144, silu |
| `dflash_config.block_size` | 8 (the kernels' minimum; a cycle drafts 8 rows) |
| `dflash_config.selector_rank` / `selector_top_k` | 256 / 16 |
| `dflash_config.mask_token_id` | 4 — Gemma4's real `<mask>` token, so no `mask_embedding.safetensors` is needed unless its quantized table loses the row |
| `dflash_config.target_layer_ids` | at most 8 ascending, e.g. `[1, 12, 24, 35, 47]` |

## Objective

1. Masked-LM head over the block: given the target's hidden state at the
   anchor position (read from `target_layer_ids`), predict the next 8 tokens
   the teacher produced — the DFlash2 block objective.
2. Selector distillation (EAGLE/DFlash-style): the selector's top-16
   candidate rows are trained toward the target's actual next tokens, so the
   target's verify step accepts as often as possible.
3. Target metric: acceptance rate per cycle against the 101 tok/s
   llama.cpp-with-MTP baseline on the same prompts; train until the
   draft's accepted tokens per cycle beats MTP's effective rate.

## Export and release

Export `config.json` plus BF16 `model.safetensors` (sharded with an index if
large), and `mask_embedding.safetensors` (one `mask_embedding` tensor of 5376)
only if the GGUF's quantized token table loses the `<mask>` row. Publish to
`incoai/Gemma-4-31B-DFlash2` on the Hugging Face Hub; the installer then
pairs it automatically (`install/families.py`), and `splash model-check`
validates the pair before any weight loads.
