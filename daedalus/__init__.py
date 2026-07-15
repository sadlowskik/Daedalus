"""Daedalus -- a from-scratch, recurrent-depth, mixture-of-experts coding LM.

A learning-first research architecture. Components are named after Greek myth,
each name describing what the piece does:

    Daedalus     the model (the master craftsman)
    Labyrinth    the shared core that loops back on itself (recurrent depth)
    Ariadne      the thread that decides how deep to go (adaptive halting)
    Muses        the routed experts (specialization emerges from data)
    Apollo       the router that picks which Muses speak
    Themis       the always-on shared experts (coding standards)
    Mnemosyne    lossy gist memory (high-level recollection)
    Scribe       exact symbol table (never approximated)
"""
from .tokenizer import ByteTokenizer
from .layers import Embeddings, Head, MultiHeadAttention, FeedForward, Block
from .models import Daedalus, Labyrinth
from .ariadne import Ariadne, ponder_loss, expected_steps
from .moe import (Expert, Router, MoELayer, MoEBlock, DaedalusMoE, load_balance_loss)
from .unified import UnifiedDaedalus
from .memory import Mnemosyne, MemoryModel, Scribe
from .rope import build_rope_cache, rotate_half, apply_rope, RoPEAttention
from .full import DaedalusFull, RoPEMoEBlock, RecurrentMoECore, MemoryLayer

__version__ = "0.2.0"

__all__ = [
    "ByteTokenizer",
    "Embeddings", "Head", "MultiHeadAttention", "FeedForward", "Block",
    "Daedalus", "Labyrinth",
    "Ariadne", "ponder_loss", "expected_steps",
    "Expert", "Router", "MoELayer", "MoEBlock", "DaedalusMoE", "load_balance_loss",
    "UnifiedDaedalus",
    "Mnemosyne", "MemoryModel", "Scribe",
    "build_rope_cache", "rotate_half", "apply_rope", "RoPEAttention",
    "DaedalusFull", "RoPEMoEBlock", "RecurrentMoECore", "MemoryLayer",
]
