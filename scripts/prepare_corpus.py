"""Build a BPE-tokenized, sharded corpus for real training runs.

`data.py` was written for the toy scale: read a directory of files, hold the
whole thing in one uint8 tensor. That caps you at roughly 100M tokens and at
one byte per token. This script is the scaled version:

  * documents come from local files **or** a streamed Hugging Face dataset,
  * they are tokenized with a byte-level BPE (~4x fewer tokens per byte),
  * output is written as uint16 `.bin` shards that training memory-maps,
  * val/test are held out **by document**, so no document spans the boundary.

Examples
--------
Natural language (phase 1), 2B tokens from FineWeb-Edu::

    python scripts/prepare_corpus.py --preset fineweb-edu \
        --out ./corpus/nl --target-tokens 2_000_000_000 --train-tokenizer

Code (phase 2), from local repos already on disk::

    python scripts/prepare_corpus.py --local ./rust_repos --ext .rs .toml .md \
        --out ./corpus/code --tokenizer ./corpus/nl/tokenizer.json

Mixed shards (recommended over a hard NL->code switch; see --mix)::

    python scripts/prepare_corpus.py --preset fineweb-edu --mix code=0.3 \
        --mix-local ./rust_repos --out ./corpus/mixed --target-tokens 5_000_000_000

The tokenizer must be **the same** across phases. Train it once, on a sample
that already contains both kinds of text, and reuse the file everywhere --
retokenizing between phases would invalidate every learned embedding.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Iterable, Iterator, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daedalus.bpe import BPETokenizer                        # noqa: E402

EOT = "<|endoftext|>"

# Streamed Hugging Face presets. These were correct when written; dataset names
# and field names do change, so check the dataset card if one 404s -- the
# --hf-* flags below cover any dataset without needing a new preset.
PRESETS = {
    "fineweb-edu":  dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                         field="text", split="train"),
    "fineweb":      dict(path="HuggingFaceFW/fineweb", name="sample-10BT",
                         field="text", split="train"),
    "cosmopedia":   dict(path="HuggingFaceTB/cosmopedia-100k", name=None,
                         field="text", split="train"),
    "tinystories":  dict(path="roneneldan/TinyStories", name=None,
                         field="text", split="train"),
    "stack-python": dict(path="bigcode/the-stack-smol", name="data/python",
                         field="content", split="train"),
    "stack-rust":   dict(path="bigcode/the-stack-smol", name="data/rust",
                         field="content", split="train"),
}


# --------------------------------------------------------------- doc sources

def iter_local(root: str, exts: Iterable[str], max_bytes: int = 2_000_000
               ) -> Iterator[str]:
    """Yield the contents of every matching file under `root`."""
    exts = tuple(exts)
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not path.endswith(exts) or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > max_bytes:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if text.strip():
            yield text


def iter_hf(path: str, name: Optional[str], split: str, field: str) -> Iterator[str]:
    """Stream a Hugging Face dataset without downloading it in full."""
    try:
        from datasets import load_dataset
    except ImportError:                                      # pragma: no cover
        raise SystemExit(
            "streaming a HF dataset needs `datasets`:  pip install datasets\n"
            "(or use --local to build from files already on disk)")
    ds = load_dataset(path, name=name, split=split, streaming=True)
    for row in ds:
        text = row.get(field)
        if text:
            yield text


def build_source(args) -> Iterator[str]:
    if args.local:
        return iter_local(args.local, args.ext)
    if args.preset:
        p = PRESETS[args.preset]
        return iter_hf(p["path"], p["name"], p["split"], p["field"])
    if args.hf_dataset:
        return iter_hf(args.hf_dataset, args.hf_config, args.hf_split, args.text_field)
    raise SystemExit("pick a source: --local, --preset, or --hf-dataset")


def interleave(primary: Iterator[str], secondary: Iterator[str], p_secondary: float,
               seed: int = 1337) -> Iterator[str]:
    """Randomly interleave two document streams.

    Mixing beats a hard phase switch: training purely on prose and *then* purely
    on code causes the model to forget the prose (catastrophic forgetting,
    McCloskey & Cohen 1989). Keeping a fraction of each throughout, and shifting
    the ratio between phases, keeps both.
    """
    rng = random.Random(seed)
    a_done = b_done = False
    while not (a_done and b_done):
        take_b = (not b_done) and (a_done or rng.random() < p_secondary)
        try:
            yield next(secondary) if take_b else next(primary)
        except StopIteration:
            if take_b:
                b_done = True
            else:
                a_done = True


# ------------------------------------------------------------------ tokenizer

def get_tokenizer(args, sample_source) -> BPETokenizer:
    if args.tokenizer and os.path.exists(args.tokenizer):
        tok = BPETokenizer.load(args.tokenizer)
        print(f"tokenizer: loaded {args.tokenizer} (vocab {tok.vocab_size:,})")
        return tok
    if not args.train_tokenizer:
        raise SystemExit(
            f"no tokenizer at {args.tokenizer!r}. Pass --train-tokenizer to learn "
            "one, or point --tokenizer at an existing file.")

    print(f"training tokenizer on ~{args.tokenizer_sample_mb}MB sample...")
    budget = args.tokenizer_sample_mb * 1_000_000
    sample, total = [], 0
    for doc in sample_source:
        sample.append(doc)
        total += len(doc)
        if total >= budget:
            break
    t0 = time.time()
    tok = BPETokenizer.train(sample, vocab_size=args.vocab_size, verbose=True)
    path = args.tokenizer or os.path.join(args.out, "tokenizer.json")
    tok.save(path)
    n_bytes = sum(len(s.encode("utf-8")) for s in sample[:200])
    n_ids = sum(len(tok.encode(s)) for s in sample[:200])
    print(f"tokenizer: {tok.vocab_size:,} tokens, trained in {time.time()-t0:.0f}s, "
          f"{n_bytes/max(n_ids,1):.2f} bytes/token -> {path}")
    return tok


# --------------------------------------------------------------------- writer

class ShardWriter:
    """Accumulates ids and flushes fixed-size uint16 shards to disk."""

    def __init__(self, out_dir: str, split: str, shard_tokens: int, dtype):
        self.out_dir, self.split = out_dir, split
        self.shard_tokens, self.dtype = shard_tokens, dtype
        self.buf: List[int] = []
        self.index = 0
        self.total = 0

    def add(self, ids: List[int]) -> None:
        self.buf.extend(ids)
        self.total += len(ids)
        while len(self.buf) >= self.shard_tokens:
            self._flush(self.buf[:self.shard_tokens])
            del self.buf[:self.shard_tokens]

    def close(self) -> None:
        if self.buf:
            self._flush(self.buf)
            self.buf = []

    def _flush(self, chunk: List[int]) -> None:
        path = os.path.join(self.out_dir, f"{self.split}_{self.index:05d}.bin")
        np.asarray(chunk, dtype=self.dtype).tofile(path)
        self.index += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--local", help="directory of source files")
    src.add_argument("--ext", nargs="+", default=[".py", ".rs", ".md", ".txt"])
    src.add_argument("--hf-dataset", help="any HF dataset id (streamed)")
    src.add_argument("--hf-config", default=None)
    src.add_argument("--hf-split", default="train")
    src.add_argument("--text-field", default="text")

    mix = ap.add_argument_group("mixing")
    mix.add_argument("--mix", type=float, default=0.0,
                     help="fraction of documents drawn from the secondary source")
    mix.add_argument("--mix-local", help="secondary source: local directory")
    mix.add_argument("--mix-preset", choices=sorted(PRESETS))

    out = ap.add_argument_group("output")
    out.add_argument("--out", required=True)
    out.add_argument("--target-tokens", type=int, default=0,
                     help="stop after roughly this many train tokens (0 = all)")
    out.add_argument("--shard-tokens", type=int, default=100_000_000)
    out.add_argument("--val-tokens", type=int, default=5_000_000)
    out.add_argument("--test-tokens", type=int, default=5_000_000)

    tk = ap.add_argument_group("tokenizer")
    tk.add_argument("--tokenizer", default=None, help="path to tokenizer.json")
    tk.add_argument("--train-tokenizer", action="store_true")
    tk.add_argument("--vocab-size", type=int, default=32768)
    tk.add_argument("--tokenizer-sample-mb", type=int, default=200)

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    args.tokenizer = args.tokenizer or os.path.join(args.out, "tokenizer.json")

    # The tokenizer sample is drawn from a *separate* pass over the source so it
    # is not consumed from the stream that gets written.
    tok = get_tokenizer(args, build_source(args))
    if tok.vocab_size > 65535:
        dtype, dtype_name = np.uint32, "uint32"
    else:
        dtype, dtype_name = np.uint16, "uint16"
    eot_id = tok.specials.get(EOT)

    docs = build_source(args)
    if args.mix > 0:
        sec_args = argparse.Namespace(**vars(args))
        sec_args.local, sec_args.preset, sec_args.hf_dataset = args.mix_local, args.mix_preset, None
        docs = interleave(docs, build_source(sec_args), args.mix, args.seed)

    writers = {s: ShardWriter(args.out, s, args.shard_tokens, dtype)
               for s in ("val", "test", "train")}
    quotas = {"val": args.val_tokens, "test": args.test_tokens}

    print(f"writing {dtype_name} shards to {args.out} "
          f"(target {args.target_tokens or 'all'} train tokens)")
    t0, n_docs = time.time(), 0
    for doc in docs:
        ids = tok.encode(doc)
        if eot_id is not None:
            ids.append(eot_id)
        # Fill the held-out splits first, then everything else is train. Because
        # splits are filled by whole document, no document is ever split across
        # the train/val boundary.
        if writers["val"].total < quotas["val"]:
            writers["val"].add(ids)
        elif writers["test"].total < quotas["test"]:
            writers["test"].add(ids)
        else:
            writers["train"].add(ids)
        n_docs += 1
        if n_docs % 2000 == 0:
            tt = writers["train"].total
            rate = tt / max(time.time() - t0, 1e-9)
            print(f"  {n_docs:,} docs | {tt/1e6:.1f}M train tokens | "
                  f"{rate/1e3:.0f}k tok/s", flush=True)
        if args.target_tokens and writers["train"].total >= args.target_tokens:
            break

    for w in writers.values():
        w.close()

    meta = {
        "vocab_size": tok.vocab_size,
        "dtype": dtype_name,
        "tokenizer": os.path.relpath(args.tokenizer, args.out),
        "eot_id": eot_id,
        "documents": n_docs,
        "tokens": {s: w.total for s, w in writers.items()},
        "shards": {s: w.index for s, w in writers.items()},
        "source": {k: v for k, v in vars(args).items()
                   if k in ("preset", "local", "hf_dataset", "hf_config", "mix",
                            "mix_local", "mix_preset", "ext")},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\ndone in {time.time()-t0:.0f}s: "
          + " | ".join(f"{s} {w.total/1e6:.1f}M" for s, w in writers.items()))
    print(f"meta -> {os.path.join(args.out, 'meta.json')}")


if __name__ == "__main__":
    main()
