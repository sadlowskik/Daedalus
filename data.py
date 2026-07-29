"""Byte-level data pipeline.

Gathers Python source files, splits them BY FILE (so no file's content leaks
across the train/val/test boundary), tokenizes with the byte tokenizer, and
saves each split as a uint8 tensor.

Usage:
    python data.py --source /usr/lib/python3.12 --out ./data
"""
from __future__ import annotations
import argparse
import bisect
import glob
import json
import os
import random
from typing import Dict

import torch

from daedalus import ByteTokenizer


def build_splits(source_dir: str, out_dir: str, max_bytes: int = 8_000_000,
                 seed: int = 1337) -> None:
    tok = ByteTokenizer()
    files, total = [], 0
    for f in sorted(glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True)):
        try:
            n = os.path.getsize(f)
        except OSError:
            continue
        if 0 < n < 200_000:
            files.append(f)
            total += n
            if total >= max_bytes:
                break

    random.seed(seed)
    random.shuffle(files)                       # deterministic

    n = len(files)
    n_val = max(1, n // 20)
    n_test = max(1, n // 20)
    splits = {
        "test": files[:n_test],
        "val": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, flist in splits.items():
        text = "\n\n".join(
            open(f, encoding="utf-8", errors="replace").read() for f in flist
        )
        ids = torch.tensor(tok.encode(text), dtype=torch.uint8)
        torch.save(ids, os.path.join(out_dir, f"{name}.pt"))
        print(f"{name:5s}: {len(flist):4d} files, {len(ids):>10,d} tokens")


def load_splits(out_dir: str) -> Dict[str, torch.Tensor]:
    return {name: torch.load(os.path.join(out_dir, f"{name}.pt"))
            for name in ("train", "val", "test")}


# --------------------------------------------------------------------------
# Sharded memory-mapped corpora (for runs too big to hold in RAM)
# --------------------------------------------------------------------------

class Corpus:
    """A split stored as one or more memory-mapped `.bin` shards.

    `load_splits` above reads the whole split into a tensor, which is fine up to
    ~100M tokens and impossible past a few billion. A memmap leaves the data on
    disk and lets the OS page in only the windows actually sampled, so corpus
    size stops being bounded by RAM. Shards are sampled in proportion to their
    length, so every token remains equally likely regardless of how the corpus
    was chunked at write time.

    Written by `scripts/prepare_corpus.py`; `meta.json` records the dtype and
    the tokenizer the ids belong to.
    """

    def __init__(self, data_dir: str, split: str):
        import numpy as np
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not paths:
            single = os.path.join(data_dir, f"{split}.bin")
            paths = [single] if os.path.exists(single) else []
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} in {data_dir}")
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in paths]
        self.lengths = [len(s) for s in self.shards]
        self.total = sum(self.lengths)
        # Cumulative lengths let a single uniform draw pick a shard in proportion
        # to its size without materialising a per-token probability vector.
        self.cum = []
        running = 0
        for n in self.lengths:
            running += n
            self.cum.append(running)

    def __len__(self) -> int:
        return self.total

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def batch(self, batch_size: int, block_size: int, device: str,
              generator: "torch.Generator | None" = None):
        """Random contiguous windows; targets are inputs shifted by one."""
        import numpy as np
        xs, ys = [], []
        for _ in range(batch_size):
            r = int(torch.randint(0, self.total, (1,), generator=generator).item())
            si = bisect.bisect_right(self.cum, r)
            shard = self.shards[si]
            hi = len(shard) - block_size - 1
            if hi <= 0:                       # shard shorter than a window
                continue
            off = int(torch.randint(0, hi, (1,), generator=generator).item())
            win = np.asarray(shard[off:off + block_size + 1], dtype=np.int64)
            xs.append(torch.from_numpy(win[:-1]))
            ys.append(torch.from_numpy(win[1:]))
        if not xs:
            raise RuntimeError("every shard is shorter than block_size + 1")
        x, y = torch.stack(xs), torch.stack(ys)
        if device.startswith("cuda"):
            # pin + non_blocking overlaps the host->device copy with compute
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)


def load_corpus(data_dir: str) -> Dict[str, Corpus]:
    """Load whichever of train/val/test exist as memmapped corpora."""
    out = {}
    for split in ("train", "val", "test"):
        try:
            out[split] = Corpus(data_dir, split)
        except FileNotFoundError:
            pass
    if "train" not in out:
        raise FileNotFoundError(f"no train shards in {data_dir}")
    return out


def get_batch(data: Dict[str, torch.Tensor], split: str, batch_size: int,
              block_size: int, device: str):
    """Random contiguous windows; targets are inputs shifted by one."""
    stream = data[split]
    ix = torch.randint(0, len(stream) - block_size - 1, (batch_size,))
    x = torch.stack([stream[i:i + block_size].long() for i in ix])
    y = torch.stack([stream[i + 1:i + block_size + 1].long() for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory of .py files")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--max-bytes", type=int, default=8_000_000)
    args = ap.parse_args()
    build_splits(args.source, args.out, args.max_bytes)
