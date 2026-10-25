function [Z, c] = gpt_forward(X, P, eps_ln)
%GPT_FORWARD Indices X (B x T, 1-based) -> logits Z ((BT) x V).
%   Embeddings, numel(P.blocks) Transformer blocks, the final LayerNorm and W_voc.
%   The B x T x d hidden states are flattened column-major, so row r = b + (t-1) B;
%   the targets must be flattened the same way, as Y(:).
[H, O] = embed_forward(X, P.W_E, P.W_P);           % B x T x d (Assignment 1)
nb = numel(P.blocks);
blocks = cell(1, nb);
for l = 1:nb
    [H, blocks{l}] = block_forward(H, P.blocks(l), eps_ln);
end
[Hf, lnf] = layernorm_forward(H, P.lnf_gamma, P.lnf_beta, eps_ln);
[B, T, d] = size(Hf);
Hbar = reshape(Hf, B*T, d);                         % (BT) x d, column-major rows
Z = Hbar * P.W_voc;                                 % (BT x d)(d x V)
c = struct('O', O, 'HL', H, 'lnf', lnf, 'Hbar', Hbar, 'B', B, 'T', T);
c.blocks = blocks;                                  % a cell inside struct() would build a struct array
end
