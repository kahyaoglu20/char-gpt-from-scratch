function g = gpt_backward(G_Z, c, P)
%GPT_BACKWARD Every parameter gradient from dL/dZ: head -> final LayerNorm ->
%   blocks L..1 -> embeddings. g has the same layout as P.
d = size(c.Hbar, 2);
G_Wvoc = c.Hbar' * G_Z;                             % (d x BT)(BT x V) = d x V
G_Hf = reshape(G_Z * P.W_voc', c.B, c.T, d);       % undo the flattening with the same rule
[G_H, G_gf, G_bf] = layernorm_backward(G_Hf, c.lnf, P.lnf_gamma);
nb = numel(P.blocks);
gb = cell(1, nb);
for l = nb:-1:1
    [G_H, gb{l}] = block_backward(G_H, c.blocks{l}, P.blocks(l));
end
[G_WE, G_WP] = embed_backward(G_H, c.O, size(P.W_P, 1));
g = struct('W_E', G_WE, 'W_P', G_WP, 'blocks', [gb{:}], 'lnf_gamma', G_gf, ...
           'lnf_beta', G_bf, 'W_voc', G_Wvoc);
end
