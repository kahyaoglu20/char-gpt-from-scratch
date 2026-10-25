"""
gpt_a2 -- MAT 451E Assignment 2: a complete character-level GPT trained and sampled from scratch.

Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Built on the Assignment 1 package `gpt_a1` (reused unchanged: autodiff Tensor, data pipeline, parameter
initialization, embeddings, LayerNorm, causal attention and their backward maps).

Modules (in the order the data flows)
    ops      the new differentiable primitives: tanh, log, target gathering, and GELU composed from them
    model    attention inside a block, feed-forward network, residual blocks, logits, cross-entropy,
             and the hand-derived backward maps of every new piece
    optim    Adam from its definition
    train    the step budget, the training loop, held-out validation, fixed-window evaluation
    sample   context cropping and categorical (or greedy) next-character sampling
    plots    the report figures

Conventions as in Assignment 1: 0-based indices, activations (B, T, d), right multiplication, float64.
"""
__version__ = "1.0"
