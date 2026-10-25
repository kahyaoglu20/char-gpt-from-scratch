"""
data.py -- corpus, character vocabulary, contiguous split, shifted windows.

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Python indices are 0-based: the vocabulary bijection is f : V -> {0,...,V-1}
and a split of length M has the legal window starts r = 0,...,M-T-1.
"""
import hashlib
from pathlib import Path

import numpy as np


def load_corpus(path):
    """Read the corpus exactly as stored: bytes -> UTF-8 text, no newline translation."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")                      # no strip(), no '\r\n' -> '\n' rewriting
    return text, raw


def sha256_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def build_vocab(text):
    """Sorted unique characters c_1 < ... < c_V and the bijection f(c) = index."""
    itos = sorted(set(text))                        # index -> character   (f^{-1})
    stoi = {c: i for i, c in enumerate(itos)}       # character -> index   (f)
    return itos, stoi


def encode(text, stoi):
    """x_i = f(s_i) for every character of the text -> int64 vector of length N."""
    return np.fromiter((stoi[c] for c in text), dtype=np.int64, count=len(text))


def decode(ids, itos):
    """s_i = f^{-1}(x_i) -> string."""
    return "".join(itos[i] for i in np.asarray(ids).ravel())


def split_contiguous(ids, num=9, den=10):
    """Chronological split: first floor(0.9 N) tokens train, the rest validation."""
    N = len(ids)
    n_tr = (num * N) // den                         # floor(0.9 N) in exact integer arithmetic
    return ids[:n_tr], ids[n_tr:], n_tr


def num_starts(M, T):
    """A length-T input needs T+1 consecutive tokens: M - T legal starts (if M > T)."""
    return max(M - T, 0)


def get_batch(split_ids, B, T, rng):
    """Sample B windows wholly inside one split.

    r_b ~ Uniform{0, ..., M-T-1},  X[b, t] = x[r_b + t],  Y[b, t] = x[r_b + t + 1].
    Returns X, Y of shape (B, T) and the starts r of shape (B,).
    """
    M = len(split_ids)
    if M <= T:
        raise ValueError(f"split of length {M} is too short for context T={T}: no legal window")
    r = rng.integers(0, M - T, size=B)              # high end exclusive -> 0..M-T-1
    t = np.arange(T)
    X = split_ids[r[:, None] + t]                   # (B, T)
    Y = split_ids[r[:, None] + t + 1]               # (B, T), shifted one step to the left
    return X, Y, r
