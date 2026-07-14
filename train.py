"""Train a Daedalus model on byte-level code data.

Prepare data first (see data.py), then e.g.:

    python train.py --model dense      --iters 3000
    python train.py --model labyrinth  --iters 3000 --variable-loops
    python train.py --model moe        --iters 3000
    python train.py --model unified    --iters 2500
    python train.py --model ariadne    --iters 3000 --beta 0.1

`--variable-loops` samples the loop count each step (anchored on the base
n_loops) so that recurrent models stay usable across depths at inference.
"""
from __future__ import annotations
import argparse
import math
import random
import time

import torch

from daedalus import (Daedalus, Labyrinth, DaedalusMoE, UnifiedDaedalus,
                      Ariadne, ponder_loss, expected_steps)
from data import load_splits, get_batch


def build_model(name: str, vocab: int, block_size: int, device: str):
    if name == "dense":
        return Daedalus(vocab, block_size=block_size).to(device)
    if name == "labyrinth":
        return Labyrinth(vocab, block_size=block_size).to(device)
    if name == "moe":
        return DaedalusMoE(vocab, block_size=block_size).to(device)
    if name == "unified":
        return UnifiedDaedalus(vocab, block_size=block_size).to(device)
    if name == "ariadne":
        return Ariadne(vocab, block_size=block_size).to(device)
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def evaluate(model, name, data, batch_size, block_size, device, args, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, "val", batch_size, block_size, device)
        if name in ("dense", "labyrinth"):
            losses.append(model(x, y)[1].item())
        elif name in ("moe", "unified"):
            losses.append(model(x, y)[1].item())
        elif name == "ariadne":
            p, logits = model(x)
            losses.append(ponder_loss(p, logits, y, args.lambda_prior, args.beta)[1].item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="labyrinth",
                    choices=["dense", "labyrinth", "moe", "unified", "ariadne"])
    ap.add_argument("--data", default="./data")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.01, help="MoE aux-loss weight")
    ap.add_argument("--beta", type=float, default=0.01, help="Ariadne ponder weight")
    ap.add_argument("--lambda-prior", type=float, default=0.2)
    ap.add_argument("--variable-loops", action="store_true")
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--out", default="./checkpoint.pt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_splits(args.data)
    vocab = 256
    model = build_model(args.model, vocab, args.block_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  params={n_params:,}  device={device}")

    best = float("inf")
    t0 = time.time()
    for it in range(args.iters + 1):
        if it % args.eval_interval == 0:
            val = evaluate(model, args.model, data, args.batch_size,
                           args.block_size, device, args)
            print(f"iter {it:5d} | val {val:.4f} | {val/math.log(2):.2f} bits/byte "
                  f"| {time.time()-t0:.0f}s")
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "iter": it, "val": val,
                            "args": vars(args)}, args.out)

        x, y = get_batch(data, "train", args.batch_size, args.block_size, device)

        if args.model in ("dense", "labyrinth"):
            if args.model == "labyrinth" and args.variable_loops:
                r = model.n_loops if random.random() < 0.5 else random.randint(2, model.n_loops * 2)
                logits = model(x, n_loops=r)[0]
                b, t, v = logits.shape
                loss = torch.nn.functional.cross_entropy(logits.view(b * t, v), y.view(b * t))
            else:
                loss = model(x, y)[1]
        elif args.model in ("moe", "unified"):
            _, ce, aux = model(x, y)
            loss = ce + args.alpha * aux
        elif args.model == "ariadne":
            p, logits = model(x)
            loss = ponder_loss(p, logits, y, args.lambda_prior, args.beta)[0]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\nbest val {best:.4f} ({best/math.log(2):.2f} bits/byte) -> {args.out}")


if __name__ == "__main__":
    main()
