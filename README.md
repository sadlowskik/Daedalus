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
- **DaedalusFull.** The whole architecture in one model: RoPE positions + MoE +
  input injection + interleaved memory (`core -> memory -> core`) + variable-loop
  recurrence. *(RoPE: Su et al. 2021; input injection: Huginn; interleaved
  memory: Block-Recurrent Transformer, Hutchins et al. 2022 / RMT.)*

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

**Flagship — `DaedalusFull` on Rust.** The fully integrated model (1.66M params,
RoPE + MoE + injection + interleaved memory + variable-loop recurrence), trained
on ~11M tokens of Rust (ripgrep, tokio, serde, clap, bat) on a single T4:
reaches **0.88 val loss (1.26 bits/byte)** in ~13 min, still descending. All 8
experts stay balanced under recurrence + interleaving; the test-time depth dial
survives (coherent generations at `r=3` and `r=5`). It generates Rust-textured
output — lifetimes, macros, `impl` blocks, byte strings — but not yet correct
code, exactly as expected at this size. *(The 1.26 bits/byte is not comparable
to the Python numbers above: Rust from a few repos is more repetitive, the model
is larger, and the context is longer.)*

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
# option A: local source files (e.g. the Python stdlib)
python data.py --source /usr/lib/python3.12 --out ./data

# option B: fetch a Rust corpus by cloning GitHub repos (needs internet + git)
python scripts/fetch_rust.py --out ./data --ext rs

# train (checkpoint-and-resume; long runs can span multiple sessions)
python train.py --model labyrinth --steps 3000 --variable-loops
python train.py --model moe       --steps 3000
python train.py --model adaptive  --n-embd 512 --core-layers 3 --steps 40000 --resume
```

Generate from a checkpoint:

```bash
python generate.py --checkpoint checkpoint.pt --model adaptive \
    --n-embd 512 --core-layers 3 --prompt "fn " --rep-pen 1.4
```

Run the test suite (the isolation checks that validate every component):

```bash
pip install pytest && pytest -q
```

## Scaling notes (honest)

A ~37M-param `DaedalusFullAdaptive` trained on ~117M tokens of Rust reaches a low
byte-level loss quickly (~0.46 bits/byte) — but early generation collapses into
whitespace. Two reasons, both worth knowing:

1. **Loss ≠ capability.** Deeply-nested code is dominated by indentation, so a
   model can drive loss down by mastering whitespace long before it learns real
   structure. Watch *generation*, not just the loss curve.
2. **Redundant data inflates the number.** Scraped repos share boilerplate,
   generated code, and near-duplicate files, so low loss partly reflects how
   predictable the data is.

Coherent code needs (a) much more training (this is <1 epoch), (b) more/cleaner
data (the full Stack, deduped), and (c) scale. See the roadmap.

## Roadmap

- [x] RoPE positions (`daedalus/rope.py`)
- [x] Input injection into the recurrent core (Huginn-style)
- [x] Integrated `DaedalusFull` + first Rust training run
- [ ] Fuse adaptive halting (Ariadne) into `DaedalusFull`
- [ ] DeepSeek-style auxiliary-loss-free load balancing
- [ ] `transformers`-compatible model class (for LoRA / vLLM ecosystem)
- [ ] Scale up the compute ladder (100M → 1B) and release weights
- [ ] Plan → execute flow (Metis → Talos) and constitution verifier (Oracle)

## License

MIT — see [LICENSE](LICENSE). Contributions and forks welcome.
