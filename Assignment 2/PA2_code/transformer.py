"""Transformer encoder, decoder and classifier, implemented from scratch.

Conventions
-----------
* B = batch, T = sequence length, C = n_embd, H = n_head, hd = C // H.
* Every model that has attention returns its attention maps as a list with one
  tensor of shape (B, T, T) per (layer, head), ordered layer 0 head 0, layer 0
  head 1, ..., so that utilities.Utilities.sanity_check can plot each of them.
* Token id 0 is <pad>.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention with H heads (optionally causal)."""

    def __init__(self, n_embd, n_head, block_size, causal=False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        if causal:
            mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
            self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, key_mask=None):
        """x: (B, T, C). key_mask: optional bool (B, T), True = real token."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        if self.causal:
            scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)  # each row sums to 1

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), attn


class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, n_embd, n_head, n_ff, block_size, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, causal=causal)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, n_ff)

    def forward(self, x, key_mask=None):
        a, attn = self.attn(self.ln1(x), key_mask)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, attn


def _split_heads(attn):
    """(B, H, T, T) -> list of H tensors (B, T, T)."""
    return [attn[:, h] for h in range(attn.shape[1])]


class TransformerEncoder(nn.Module):
    """Token + learned position embeddings -> N bidirectional blocks.

    forward(x) returns (hidden (B, T, C), attn_maps). Padding (id 0) is masked
    out of the attention keys.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, n_ff, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, n_ff, block_size, causal=False) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        key_mask = x != self.pad_id
        # Guard against an all-padding row (would give NaN in softmax).
        key_mask = key_mask | (~key_mask.any(dim=1, keepdim=True))
        attn_maps = []
        for blk in self.blocks:
            h, attn = blk(h, key_mask)
            attn_maps.extend(_split_heads(attn))
        return self.ln_f(h), attn_maps


class FeedforwardClassifier(nn.Module):
    """One hidden layer: n_input -> n_hidden -> n_output."""

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.net(x)


class EncoderClassifier(nn.Module):
    """Encoder + mean pooling (over non-pad positions) + feedforward classifier.

    forward(x) returns logits (B, n_output), so it can be trained end to end
    with cross-entropy. The encoder is exposed as self.encoder (utilities.py
    needs it to get attention maps).
    """

    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        h, _ = self.encoder(x)  # (B, T, C)
        mask = (x != self.encoder.pad_id).unsqueeze(-1).to(h.dtype)  # (B, T, 1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.classifier(pooled)


class TransformerDecoderLM(nn.Module):
    """GPT-like word-level decoder with causal self-attention.

    forward(x)            -> (logits (B, T, V), attn_maps)
    forward(x, targets)   -> scalar cross-entropy loss (mean over all positions)
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, n_ff):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, n_ff, block_size, causal=True) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        attn_maps = []
        for blk in self.blocks:
            h, attn = blk(h)
            attn_maps.extend(_split_heads(attn))
        logits = self.lm_head(self.ln_f(h))
        if targets is None:
            return logits, attn_maps
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
