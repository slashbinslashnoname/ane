#!/usr/bin/env python3
"""Fetch tokenizer + wiki_text dataset, tokenize to uint16 .bin for train_large.

Pipeline (idempotent — only does work that's missing):
  1. Download Llama2 SentencePiece tokenizer (tokenizer.model)
  2. Convert it to llama2.c-format tokenizer.bin (used by dashboard.py)
  3. Stream yoandrey/wiki_text from HuggingFace
  4. Tokenize (BOS + text + EOS per doc), write flat uint16 -> train_data00.bin

Env:
  MAX_TOKENS=N            cap output token count (default 50_000_000)
  DATASET=name            override HF dataset (default yoandrey/wiki_text)
"""
import os, sys, struct, subprocess, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent.parent / 'assets' / 'models'
TOKENIZER_MODEL = ASSETS / 'tokenizer.model'
TOKENIZER_BIN = ASSETS / 'tokenizer.bin'
DATA_BIN = ROOT / 'train_data00.bin'

TOKENIZER_URL = 'https://github.com/karpathy/llama2.c/raw/master/tokenizer.model'
DATASET = os.environ.get('DATASET', 'yoandrey/wiki_text')

MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 50_000_000))
BATCH_SIZE = 4096


def pip_install(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *pkgs])


def ensure_deps():
    needed = []
    for mod, pkg in [('sentencepiece', 'sentencepiece'),
                     ('datasets', 'datasets'),
                     ('tqdm', 'tqdm')]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f"Installing python deps: {', '.join(needed)}")
        pip_install(*needed)


def download(url, dst, label=None):
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    label = label or dst.name
    print(f"Downloading {label} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + '.part')
    last = [0]
    def hook(blocks, bsize, total):
        if total <= 0:
            return
        done = blocks * bsize
        pct = min(100, done * 100 // total)
        if pct != last[0]:
            last[0] = pct
            sys.stdout.write(f"\r  {pct:3d}% ({done/1e6:.1f}/{total/1e6:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.rename(dst)


def export_tokenizer_bin():
    if TOKENIZER_BIN.exists():
        return
    import sentencepiece as spm
    print(f"Building {TOKENIZER_BIN.name}")
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    tokens, scores = [], []
    for i in range(sp.vocab_size()):
        t = sp.id_to_piece(i)
        s = sp.get_score(i)
        if i == sp.bos_id():
            t = '\n<s>\n'
        elif i == sp.eos_id():
            t = '\n</s>\n'
        t = t.replace('▁', ' ')
        tokens.append(t.encode('utf-8'))
        scores.append(s)
    max_len = max(len(t) for t in tokens)
    with open(TOKENIZER_BIN, 'wb') as f:
        f.write(struct.pack('i', max_len))
        for tok, score in zip(tokens, scores):
            f.write(struct.pack('f', score))
            f.write(struct.pack('i', len(tok)))
            f.write(tok)


TEXT_KEYS = ('text', 'content', 'body', 'article', 'page', 'document')


def extract_text(row):
    """Find the text field in an HF row (handles common schemas)."""
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: first string field
    for v in row.values():
        if isinstance(v, str) and len(v.strip()) > 32:
            return v.strip()
    return ''


def tokenize_dataset():
    if DATA_BIN.exists() and DATA_BIN.stat().st_size > 0:
        n = DATA_BIN.stat().st_size // 2
        print(f"{DATA_BIN.name} already exists ({n:,} tokens, {DATA_BIN.stat().st_size/1e6:.1f} MB)")
        return
    import sentencepiece as spm
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading {DATASET} (streaming)")
    ds = load_dataset(DATASET, split='train', streaming=True)
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    eos = sp.eos_id()

    tmp = DATA_BIN.with_suffix('.bin.part')
    total = 0
    skipped = 0
    batch = []
    pbar = tqdm(total=MAX_TOKENS, unit='tok', unit_scale=True, desc='Tokenizing')

    def flush(out):
        nonlocal total
        if not batch:
            return
        id_lists = sp.encode(batch, add_bos=True)
        for ids in id_lists:
            ids.append(eos)
            out.write(struct.pack(f'<{len(ids)}H', *ids))
            total += len(ids)
            pbar.update(len(ids))
        batch.clear()

    with open(tmp, 'wb') as out:
        for row in ds:
            text = extract_text(row)
            if not text:
                skipped += 1
                continue
            batch.append(text)
            if len(batch) >= BATCH_SIZE:
                flush(out)
                if total >= MAX_TOKENS:
                    break
        flush(out)
    pbar.close()
    tmp.rename(DATA_BIN)
    print(f"Wrote {total:,} tokens -> {DATA_BIN} ({DATA_BIN.stat().st_size/1e6:.1f} MB) "
          f"[skipped {skipped} empty rows]")


def main():
    ensure_deps()
    download(TOKENIZER_URL, TOKENIZER_MODEL, 'tokenizer.model (Llama2 BPE)')
    export_tokenizer_bin()
    tokenize_dataset()
    print("\nReady. Run `make train` to start training.")


if __name__ == '__main__':
    main()
