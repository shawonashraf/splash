#!/usr/bin/env python3
"""Train a DFlash2 draft for Gemma4-31B on captured teacher activations.

The draft matches the engine's contract (runtime/model/DraftCheckpoint.cpp,
runtime/metal/abi/DraftAttention.h): one layer of hidden 5376 over the
concatenated captured hidden states of the target layers, grouped dynamic
convolutions (stages 2, taps 2, group 16), 32 query / 8 KV heads of 128, a
2048-token window over the context, the target's own lm_head for proposals,
and a rank-256 bilinear candidate selector.

Objective per anchor position: cross-entropy of the block's 8 rows against
the teacher's next 8 tokens (per-position decay gamma^-j), plus a selector
transition loss over the top-16 unary candidates.

Usage:
  .venv/bin/python dev/draft/gemma4/train.py \
      --captures /tmp/g4-captures-full --teacher ~/Models/.../MLX-8bit \
      --out /tmp/g4-draft --steps 2000 --batch 4 --ctx 2048
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 5376
HEAD_DIM = 128
Q_HEADS = 32
KV_HEADS = 8
ATTN_W = Q_HEADS * HEAD_DIM
INTER = 6144
BLOCK = 8
RANK = 256
TOPK = 16
TAPS = 2
STAGES = 2
GROUP = 16
GROUPS = H // GROUP
VOCAB = 262144
ROPE_THETA = 10_000.0


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def conv_prepare(conv, x):
    """x: [BLOCK, H] -> (conv output, delta kernel of the other stage)."""
    kernels = conv.kernel_projection(x).view(BLOCK, STAGES, TAPS, GROUPS)
    delta = kernels[:, 0].reshape(BLOCK, TAPS, GROUPS)
    base = conv.base_kernel[0]
    flat = x.view(BLOCK, GROUPS, GROUP)
    out = torch.zeros_like(flat)
    pos = torch.arange(BLOCK, device=x.device)
    for tap in range(TAPS):
        shifted = flat if tap == 0 else torch.cat([flat.new_zeros(tap, GROUPS, GROUP), flat[:-tap]], dim=0)
        coef = base[tap].view(GROUPS, GROUP) + delta[:, tap].unsqueeze(-1)
        valid = (pos >= tap).view(BLOCK, 1, 1).to(x.dtype)
        out = out + shifted * coef * valid
    return out.view(BLOCK, H), kernels[:, 1]


def conv_finish(conv, x, delta_kernel):
    base = conv.base_kernel[1]
    flat = x.view(BLOCK, GROUPS, GROUP)
    d = delta_kernel.reshape(BLOCK, TAPS, GROUPS)
    out = torch.zeros_like(flat)
    pos = torch.arange(BLOCK, device=x.device)
    for tap in range(TAPS):
        shifted = flat if tap == 0 else torch.cat([flat.new_zeros(tap, GROUPS, GROUP), flat[:-tap]], dim=0)
        coef = base[tap].view(GROUPS, GROUP) + d[:, tap].unsqueeze(-1)
        valid = (pos >= tap).view(BLOCK, 1, 1).to(x.dtype)
        out = out + shifted * coef * valid
    return out.view(BLOCK, H)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_kernel = nn.Parameter(torch.zeros(STAGES, TAPS, H))
        with torch.no_grad():
            self.base_kernel[:, 0].fill_(1.0)
        self.kernel_projection = nn.Linear(H, STAGES * TAPS * GROUPS, bias=False)
        with torch.no_grad():
            self.kernel_projection.weight.zero_()


class DraftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.attention_conv = Conv()
        self.q_proj = nn.Linear(H, ATTN_W, bias=False)
        self.k_proj = nn.Linear(H, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(ATTN_W, H, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp_conv = Conv()
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)

    def forward(self, block, ctx_k, ctx_v, ctx_pos, block_pos):
        # block: [BLOCK, H]; ctx_k/v: [C, KV_HEADS, HEAD_DIM] (k roped+normed)
        residual = block
        x = self.input_layernorm(block)
        prepared, out_kernel = conv_prepare(self.attention_conv, x)
        q = self.q_proj(prepared).view(BLOCK, Q_HEADS, HEAD_DIM).transpose(0, 1)
        k = self.k_proj(prepared).view(BLOCK, KV_HEADS, HEAD_DIM).transpose(0, 1)
        v = self.v_proj(prepared).view(BLOCK, KV_HEADS, HEAD_DIM).transpose(0, 1)
        q = rms_norm(q, self.q_norm.weight, self.q_norm.eps)
        k = rms_norm(k, self.k_norm.weight, self.k_norm.eps)
        k_all = torch.cat([ctx_k, k], dim=1)
        v_all = torch.cat([ctx_v, v], dim=1)
        q = apply_rope(q, block_pos)
        k_all = apply_rope(k_all, torch.cat([ctx_pos, block_pos]))
        scores = torch.einsum("hqd,knd->hqk", q.float(), k_all.float()) / math.sqrt(HEAD_DIM)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,knd->hqd", attn, v_all.float()).transpose(0, 1).reshape(BLOCK, ATTN_W)
        out = self.o_proj(out)
        out = conv_finish(self.attention_conv, out, out_kernel)
        hidden = residual + out

        residual = hidden
        x = self.post_attention_layernorm(hidden)
        prepared, out_kernel = conv_prepare(self.mlp_conv, x)
        inter = F.silu(self.gate_proj(prepared).float()) * self.up_proj(prepared).float()
        down = self.down_proj(inter)
        down = conv_finish(self.mlp_conv, down, out_kernel)
        return residual + down


def rope_tables(positions):
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=positions.device).float() / HEAD_DIM))
    ang = positions.float()[:, None] * freqs[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, positions):
    # x: [heads, N, dim]
    cos, sin = rope_tables(positions)
    cos = cos[None, :, :].to(x.dtype)
    sin = sin[None, :, :].to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class Draft(nn.Module):
    def __init__(self, k_layers):
        super().__init__()
        self.fc = nn.Linear(k_layers * H, H, bias=False)
        self.hidden_norm = RMSNorm(H)
        self.layer = DraftLayer()
        self.norm = RMSNorm(H)
        self.selector_hidden = nn.Linear(H, RANK, bias=False)
        self.predecessor_codebook = nn.Parameter(torch.empty(VOCAB, RANK))
        self.successor_codebook = nn.Parameter(torch.empty(VOCAB, RANK))
        self.mask_embedding = nn.Parameter(torch.zeros(H))
        nn.init.normal_(self.predecessor_codebook, std=0.02)
        nn.init.normal_(self.successor_codebook, std=0.02)

    def forward(self, anchor_cap, ctx_caps, ctx_pos, block_pos, lm_head):
        ctx_hidden = rms_norm(self.fc(ctx_caps), self.hidden_norm.weight, self.hidden_norm.eps)
        ctx_k = self.layer.k_proj(ctx_hidden).view(-1, KV_HEADS, HEAD_DIM).transpose(0, 1)
        ctx_v = self.layer.v_proj(ctx_hidden).view(-1, KV_HEADS, HEAD_DIM).transpose(0, 1)
        ctx_k = rms_norm(ctx_k, self.layer.k_norm.weight, self.layer.k_norm.eps)
        ctx_k = apply_rope(ctx_k, ctx_pos)
        block = torch.cat([self.fc(anchor_cap)[None, :],
                           self.mask_embedding[None, :].expand(BLOCK - 1, H)], dim=0)
        hidden = self.layer(block, ctx_k, ctx_v, ctx_pos, block_pos)
        final = rms_norm(hidden, self.norm.weight, self.norm.eps)
        logits = final.float() @ lm_head.t()  # [BLOCK, VOCAB]
        sel_hidden = self.selector_hidden(final)  # [BLOCK, RANK]
        return logits, sel_hidden


def load_lm_head(teacher_dir, device):
    """The teacher's tied embedding, dequantized, as the draft's frozen head."""
    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(teacher_dir)
    emb = model.language_model.model.embed_tokens
    if hasattr(emb, "scales"):  # affine 8-bit quantized, group 64
        codes = np.array(emb.weight)  # [vocab, H/4] uint32, 4 codes per word
        scales = np.array(emb.scales.astype(mx.float32))  # [vocab, H/64]
        biases = np.array(emb.biases.astype(mx.float32))
        packed = codes.astype(np.uint64)
        vals = np.stack([(packed >> (8 * i)) & 0xFF for i in range(4)], axis=-1)  # [vocab, H/4, 4]
        vals = vals.reshape(codes.shape[0], -1).astype(np.float32)  # [vocab, H]
        groups = vals.reshape(codes.shape[0], -1, 64)
        w = groups * scales[:, :, None] + biases[:, :, None]
        arr = w.reshape(codes.shape[0], -1)
    else:
        arr = np.array(emb.weight.astype(mx.float32))
    return torch.from_numpy(arr.copy()).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--selector-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps")

    cap_dir = Path(args.captures)
    meta = json.loads((cap_dir / "meta.json").read_text())
    k_layers = len(meta["layers"])
    total = meta["tokens"]
    assert meta["hidden"] == H
    bounds = np.load(cap_dir / "bounds.i64.npy")
    captures = np.memmap(cap_dir / "captures.bf16", dtype=np.uint16, mode="r", shape=(total, k_layers * H))
    labels = np.memmap(cap_dir / "labels.i32", mode="r", shape=(total,))

    def cap_rows(rows):
        return torch.from_numpy((captures[rows].astype(np.uint32) << 16).view(np.float32).copy())

    print(f"captures: {total} tokens, {k_layers} layers", flush=True)
    print("loading teacher lm_head...", flush=True)
    lm_head = load_lm_head(args.teacher, device)
    lm_head.requires_grad_(False)
    print(f"lm_head {tuple(lm_head.shape)}", flush=True)

    model = Draft(k_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seq_starts = bounds[:-1]
    seq_ends = bounds[1:]
    w = torch.tensor([args.gamma ** (-j) for j in range(BLOCK)], device=device, dtype=torch.float32)
    w = w / w.sum() * BLOCK

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss_sum = 0.0
        for _ in range(args.batch):
            si = int(rng.integers(0, len(seq_starts)))
            lo, hi = int(seq_starts[si]), int(seq_ends[si]) - BLOCK - 1
            if hi <= lo + 1:
                continue
            t = int(rng.integers(lo + 1, hi))
            ctx_len = min(t - lo, args.ctx)
            ctx_pos = torch.arange(t - ctx_len, t, device=device)
            block_pos = torch.arange(t, t + BLOCK, device=device)
            ctx_caps = cap_rows(np.arange(t - ctx_len, t)).to(device)
            anchor_cap = cap_rows(np.array([t]))[0].to(device)
            target = torch.from_numpy(labels[t:t + BLOCK].astype(np.int64)).to(device)
            prev = torch.from_numpy(
                np.concatenate([labels[max(lo, t - 1):t], labels[t:t + BLOCK - 1]]).astype(np.int64)
            ).to(device)
            logits, sel_hidden = model(anchor_cap, ctx_caps, ctx_pos, block_pos, lm_head)
            ce = F.cross_entropy(logits, target, reduction="none") * w
            top_ids = logits.topk(TOPK, dim=-1).indices  # [BLOCK, TOPK]
            pred_emb = model.predecessor_codebook[prev]  # [BLOCK, RANK]
            succ_emb = model.successor_codebook[top_ids]  # [BLOCK, TOPK, RANK]
            transition = torch.einsum("br,bkr->bk", pred_emb * sel_hidden, succ_emb)
            scores = logits.gather(-1, top_ids) + transition
            hit = (top_ids == target[:, None]).float().argmax(-1)
            sel_ce = F.cross_entropy(scores, hit, reduction="none") * w
            loss = (ce + args.selector_alpha * sel_ce).mean() / args.batch
            loss.backward()
            loss_sum += loss.item() * args.batch
        opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss {loss_sum / args.batch:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"state": model.state_dict(), "k_layers": k_layers, "meta": meta},
                       out / f"checkpoint-{step}.pt")
            print(f"saved checkpoint-{step}.pt", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
