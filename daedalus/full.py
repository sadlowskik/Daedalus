"""DaedalusFull -- the fully integrated architecture.

Combines, in one model:
  - RoPE positions (rope.RoPEAttention)
  - fine-grained MoE (moe.MoELayer: Muses / Apollo / Themis)
  - input injection (the original embedding is re-added every loop)
  - interleaved memory (memory.Mnemosyne between recurrent stages)
  - variable-loop recurrent depth

Structure:
    Prelude(token emb)
      -> RecurrentMoECore (looped, input injection)
      -> MemoryLayer      (compress -> read back)
      -> RecurrentMoECore
      -> Coda(norm + head)

Adaptive halting (Ariadne) is intentionally NOT fused here: PonderNet's
per-step output weighting does not compose cleanly with the interleaved
core -> memory -> core stack. Use it as a separate model when you want
per-token adaptive depth.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPEAttention
from .moe import MoELayer, load_balance_loss
from .memory import Mnemosyne


class RoPEMoEBlock(nn.Module):
    """Pre-norm block: RoPE attention + MoE feed-forward."""

    def __init__(self, n_embd, n_head, block_size, n_experts, top_k, n_shared, hidden):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_head, block_size)
        self.moe = MoELayer(n_embd, n_experts, top_k, n_shared, hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, scores = self.moe(self.ln2(x))
        return x + moe_out, scores


class RecurrentMoECore(nn.Module):
    """A shared stack of MoE blocks, looped with input injection.

    The prelude embedding `e` is re-added before every loop so deep recurrence
    stays anchored to the input (Huginn, Geiping et al. 2025).
    """

    def __init__(self, n_embd, n_head, block_size, core_layers, n_experts, top_k, n_shared, hidden):
        super().__init__()
        self.blocks = nn.ModuleList([
            RoPEMoEBlock(n_embd, n_head, block_size, n_experts, top_k, n_shared, hidden)
            for _ in range(core_layers)
        ])

    def forward(self, x: torch.Tensor, e: torch.Tensor, n_loops: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux = x.new_zeros(())
        for _ in range(n_loops):
            x = x + e                                      # input injection
            for blk in self.blocks:
                x, scores = blk(x)
                aux = aux + load_balance_loss(
                    scores, scores.topk(blk.moe.top_k, -1)[1], blk.moe.n_experts)
        return x, aux


class MemoryLayer(nn.Module):
    """Interleaved memory: compress the running state into gist vectors, then
    let every position read from them via cross-attention."""

    def __init__(self, n_embd, n_gist, n_head):
        super().__init__()
        self.compress = Mnemosyne(n_embd, n_gist, n_head)
        self.read = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gist = self.compress(x)
        readout, _ = self.read(x, gist, gist)
        return x + self.ln(readout)


class DaedalusFull(nn.Module):
    """The whole architecture in one model.

    forward() returns (logits, ce_loss, aux_loss). Train with
    `ce_loss + alpha * aux_loss` and a variable loop count (sample `n_loops`
    each step) so the test-time depth dial stays usable.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 block_size: int = 256, core_layers: int = 2, n_loops: int = 3,
                 n_experts: int = 8, top_k: int = 2, n_shared: int = 1,
                 hidden: Optional[int] = None, n_gist: int = 16, n_stages: int = 2):
        super().__init__()
        hidden = hidden or n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)      # RoPE handles position
        self.stages = nn.ModuleList([
            RecurrentMoECore(n_embd, n_head, block_size, core_layers,
                             n_experts, top_k, n_shared, hidden)
            for _ in range(n_stages)
        ])
        self.memories = nn.ModuleList([
            MemoryLayer(n_embd, n_gist, n_head) for _ in range(n_stages - 1)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_loops, self.block_size = n_loops, block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                n_loops: Optional[int] = None):
        r = n_loops if n_loops is not None else self.n_loops
        e = self.tok_emb(idx)
        x = e
        aux = x.new_zeros(())
        for i, stage in enumerate(self.stages):
            x, a = stage(x, e, r)
            aux = aux + a
            if i < len(self.memories):
                x = self.memories[i](x)                      # interleaved memory
        logits = self.lm_head(self.ln_f(x))
        ce = None
        if targets is not None:
            b, t, v = logits.shape
            ce = F.cross_entropy(logits.view(b * t, v), targets.view(b * t))
        return logits, ce, aux
