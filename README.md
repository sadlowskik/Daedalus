# Daedalus

A small, **recurrent-depth, mixture-of-experts** language model for code, built
from scratch as a learning-first research project. The goal is not to beat
frontier models — it is to understand, and to make every mechanism inspectable,
hackable, and honestly measured.

Every component is named after Greek myth, and each name describes what the
piece does:

| Name | Component | What it does |
|------|-----------|--------------|
| **Daedalus** | the model | the master craftsman |
| **Labyrinth** | recurrent-depth core | a shared block looped back on itself |
| **Ariadne** | adaptive halting | decides how deep to loop, per token |
| **Muses** | routed experts | specialization emerges from data |
| **Apollo** | the router | picks which Muses speak |
| **Themis** | shared experts | always-on, carry the common ground |
| **Mnemosyne** | gist memory | lossy, high-level recollection |
| **Scribe** | symbol table | exact, never approximated |

## Why this architecture

- **Recurrent depth (Labyrinth).** Loop one shared core `r` times to get the
  effective depth of `r` layers at the parameter cost of one. Decouples *how
  much the model computes* from *how big it is* — ideal when memory, not time,
  is the bottleneck. *(Universal Transformer; Huginn, Geiping et al. 2025; Ouro.)*
- **Adaptive halting (Ariadne).** A PonderNet halting head lets each token
  choose its own depth — more loops on hard tokens, fewer on easy ones.
  *(PonderNet, Banino et al. 2021; ACT, Graves 2016.)*
- **Fine-grained MoE (Muses / Apollo / Themis).** Many small experts, a noisy
  top-k router, and an always-on shared expert. More capacity at similar active
  compute. *(DeepSeekMoE; Switch Transformer; Shazeer et al. 2017.)*
- **Two-tier memory (Mnemosyne + Scribe).** Compress fuzzy context lossily, but
  keep identifiers/signatures/paths bit-exact in an AST-parsed symbol table —
  because a single hallucinated identifier breaks compilation.
- **Unified (Mixture-of-Recursions).** Loop a *shared MoE core*: recurrent depth
  and sparse experts at once. *(Bae et al. 2025.)*

## Results (toy scale)

Byte-level, ~0.68M–0.8M params, trained on the CPython standard library on a
single T4 GPU. These are **learning-scale** numbers — reported honestly, not to
impress:

| Model | Val loss | bits/byte | Note |
|-------|---------:|----------:|------|
| Daedalus (dense, 3 layers) | 1.32 | 1.91 | baseline |
| **Labyrinth** (3-layer core × 4 loops) | **1.19** | **1.72** | beats dense at **equal params** |
| DaedalusMoE (3 MoE blocks) | 1.30 | 1.87 | no expert collapse |
| UnifiedDaedalus (MoE core × 4 loops) | 1.35 | 1.95 | stable fusion (underfit) |

- **Ariadne** learns genuine per-token depth allocation (depth std ≈ 0.70;
  `corr(depth, difficulty) ≈ +0.12` — real but weak at this scale).
- **Mnemosyne** memory helps: predicting a segment with the compressed gist of
  the previous 128 tokens beats predicting it without, by ~0.39 nats.

**Honest scope:** at this size, expert and depth specialization is *structural*
(whitespace, case, punctuation), not *semantic*. Semantic specialization needs
scale. This repo is for understanding the mechanisms and as a base to scale up.

## Install

```bash
pip install torch
git clone <your-fork-url> && cd daedalus
```

## Quickstart

```python
import torch
from daedalus import Labyrinth, ByteTokenizer

tok = ByteTokenizer()
model = Labyrinth(vocab_size=256, n_embd=128, core_layers=3, n_loops=4, block_size=128)

ids = torch.tensor([tok.encode("def add(a, b):")])
logits, _ = model(ids)                    # (1, T, 256)
logits, _ = model(ids, n_loops=8)         # think deeper at inference (train variable-loops first)
```

Prepare data and train:

```bash
python data.py  --source /usr/lib/python3.12 --out ./data
python train.py --model labyrinth --iters 3000 --variable-loops
python train.py --model moe       --iters 3000
python train.py --model unified   --iters 2500
```

Run the test suite (the isolation checks that validate every component):

```bash
pip install pytest && pytest -q
```

## Roadmap

- [ ] RoPE positions (currently learned-absolute)
- [ ] Input injection into the recurrent core (Huginn-style)
- [ ] DeepSeek-style auxiliary-loss-free load balancing
- [ ] `transformers`-compatible model class (for LoRA / vLLM ecosystem)
- [ ] Scale up the compute ladder (100M → 1B) and release weights
- [ ] Plan → execute flow (Metis → Talos) and constitution verifier (Oracle)

## License

MIT — see [LICENSE](LICENSE). Contributions and forks welcome.
