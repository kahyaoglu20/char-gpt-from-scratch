"""
gpt_a1 -- MAT 451E Assignment 1: foundations of a character-level GPT, from scratch.

Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Modules (import order follows the data flow)
    autograd  minimal reverse-mode automatic differentiation on NumPy tensors
    data      corpus loading, sorted character vocabulary, contiguous split, shifted windows
    params    initialization and counting of every trainable tensor of the L-block GPT
    model     differentiable embeddings, LayerNorm, single-head causal attention (+ backward maps)
    checks    centred finite differences and error measures

Conventions shared by every module: 0-based indices, activations laid out batch-first as (B, T, d),
matrices applied by right multiplication (Y = X W, weights stored d_in x d_out), IEEE double precision.
"""
__version__ = "1.0"
