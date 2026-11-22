#!/usr/bin/env python3
"""Capture teacher hidden states for DFlash2 draft training (Gemma4-31B).

Runs the teacher (MLX checkpoint) teacher-forced over a coding corpus and
records, per token position:
  - the concatenated output hidden states of the target layers the draft
    reads (`--layers`, ascending; the draft's `fc` maps their concatenation
    to its hidden size), as bf16, and
  - the teacher's greedy next-token id (argmax of the logits at the
    position), as int32 labels.

Output (in --out):
  captures.bf16   [T, len(layers) * hidden] bf16 memmap
  labels.i32      [T] int32 memmap (token id that follows each position)
  bounds.i64      [S + 1] int64 memmap (sequence start offsets into T)
  meta.json       layers, hidden, vocab, corpus stats

Usage:
  .venv/bin/python dev/draft/gemma4/capture.py \
      --model ~/Models/lmstudio-community/gemma-4-31B-it-MLX-8bit \
      --out /tmp/g4-captures --layers 11 29 47 --seq-len 4096 --max-tokens 400000
"""

import argparse
import json
import os
import random
from pathlib import Path

import mlx.core as mx
import numpy as np


def gather_corpus(roots, max_files=4000):
    """Python source files from the given roots, as (path, text) pairs."""
    files = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules", "build", ".venv")]
            for name in filenames:
                if name.endswith((".py", ".rs", ".mm", ".cpp", ".ts", ".swift")):
                    files.append(Path(dirpath) / name)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break
    random.shuffle(files)
    out = []
    for path in files[:max_files]:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if len(text) > 20:
            out.append((str(path), text))
    return out


def synthetic_tool_transcripts(count=300):
    """Short tool-calling conversations rendered in the model's chat format."""
    rng = random.Random(1234)
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "edit_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}}}},
    ]
    snippets = [
        ("fix the off-by-one in the tokenizer", "read_file", {"path": "src/tokenizer.py"}),
        ("make the retry backoff exponential", "edit_file", {"path": "src/client.py", "old": "sleep(1)", "new": "sleep(2 ** attempt)"}),
        ("check the tests pass", "run_tests", {"path": "tests/"}),
        ("refactor the cache eviction policy", "read_file", {"path": "src/cache.rs"}),
    ]
    out = []
    for i in range(count):
        task, tool, args = rng.choice(snippets)
        msgs = [
            {"role": "user", "content": f"Please {task}. Work step by step and use tools."},
            {"role": "assistant", "content": f"I'll {task}. Let me look at the code first.",
             "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": "ok: 200 lines returned"},
            {"role": "assistant", "content": f"Done with {task}; the change is minimal and covered by tests."},
        ]
        out.append(msgs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 29, 47])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=400_000)
    ap.add_argument("--roots", nargs="*", default=["/Users/shawon/Projects", "/tmp/speculators"])
    ap.add_argument("--transcripts", type=int, default=300)
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    print("loading teacher...", flush=True)
    model, tokenizer = load(args.model)
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    import json as _json
    cfg = _json.load(open(os.path.join(args.model, "config.json")))
    hidden = cfg["text_config"]["hidden_size"]
    vocab = cfg["text_config"]["vocab_size"]
    print(f"hidden={hidden} vocab={vocab} layers={len(layers)}", flush=True)

    # Build corpus token sequences.
    seqs = []
    bounds = [0]
    total = 0
    corpus = gather_corpus(args.roots)
    print(f"corpus files: {len(corpus)}", flush=True)
    for path, text in corpus:
        ids = tokenizer.encode(text)
        for start in range(0, max(0, len(ids) - 1), args.seq_len):
            chunk = ids[start:start + args.seq_len]
            if len(chunk) < 64:
                continue
            seqs.append(chunk)
            total += len(chunk)
            bounds.append(total)
            if total >= args.max_tokens:
                break
        if total >= args.max_tokens:
            break
    # Synthetic tool-call transcripts in the chat template.
    for msgs in synthetic_tool_transcripts(args.transcripts):
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        ids = tokenizer.encode(text)
        for start in range(0, max(0, len(ids) - 1), args.seq_len):
            chunk = ids[start:start + args.seq_len]
            if len(chunk) < 64:
                continue
            seqs.append(chunk)
            total += len(chunk)
            bounds.append(total)
            if total >= args.max_tokens:
                break
        if total >= args.max_tokens:
            break
    total = bounds[-1]
    print(f"sequences: {len(seqs)} tokens: {total}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = len(args.layers)
    cap_path = out / "captures.bf16"
    lab_path = out / "labels.i32"
    bnd_path = out / "bounds.i64"
    captures = np.memmap(cap_path, dtype=np.uint16, mode="w+", shape=(total, k * hidden))
    labels = np.memmap(lab_path, dtype=np.int32, mode="w+", shape=(total,))
    bounds_np = np.array(bounds, dtype=np.int64)
    np.save(bnd_path, bounds_np)

    # Wrap the target layers to record their output hidden states.
    import mlx.nn as nn
    captured = {}

    class Capture(nn.Module):
        def __init__(self, layer, idx):
            super().__init__()
            self.layer = layer
            self.idx = idx

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("layer"), name)

        def __call__(self, *args, **kwargs):
            out = self.layer(*args, **kwargs)
            captured[self.idx] = out[0] if isinstance(out, tuple) else out
            return out

    originals = {}
    for i in args.layers:
        originals[i] = layers[i]
        layers[i] = Capture(layers[i], i)

    pos = 0
    for si, chunk in enumerate(seqs):
        if pos >= total:
            break
        n = min(len(chunk), total - pos)
        ids = mx.array([chunk[:n]])
        logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_ids = mx.argmax(logits[0, :-1], axis=-1)  # predicts token t+1
        # Captures: concat layer outputs along hidden dim.
        parts = [captured[i][0, : n - 1] for i in args.layers]
        cap = mx.concatenate(parts, axis=-1)  # [n-1, k*H]
        cap32 = np.array(cap.astype(mx.float32), dtype=np.float32)
        captures[pos:pos + n - 1] = (cap32.view(np.uint32) >> 16).astype(np.uint16)
        labels[pos:pos + n - 1] = np.array(next_ids.astype(mx.int32), dtype=np.int32)
        pos += n - 1
        if si % 20 == 0:
            print(f"  {pos}/{total} tokens captured", flush=True)
        del logits, cap, parts, next_ids
        mx.clear_cache()

    for i, layer in originals.items():
        layers[i] = layer
    del captures, labels
    meta = {
        "layers": args.layers,
        "hidden": hidden,
        "vocab": vocab,
        "tokens": int(pos),
        "sequences": len(seqs),
        "seq_len": args.seq_len,
        "model": args.model,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {pos} tokens -> {out}", flush=True)


if __name__ == "__main__":
    main()
