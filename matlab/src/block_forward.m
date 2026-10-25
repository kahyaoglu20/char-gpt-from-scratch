function [Hn, c] = block_forward(H, blk, eps_ln)
%BLOCK_FORWARD One pre-normalization Transformer block on a B x T x d tensor:
%   U = LN1(H),  F = attention(U),  R = H + F,  D = LN2(R),  G = FFN(D),  H' = R + G.
%   blk holds W_QKV, W_O, W_1, b_1, W_2, b_2 and the two LayerNorm gain/shift pairs.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
[U, ln1] = layernorm_forward(H, blk.ln1_gamma, blk.ln1_beta, eps_ln);
[F, att] = attention_forward(U, blk.W_QKV, blk.W_O);         % Assignment 1, unchanged
R = H + F;                                                     % first residual
[D, ln2] = layernorm_forward(R, blk.ln2_gamma, blk.ln2_beta, eps_ln);
[G, ffn] = ffn_forward(D, blk.W_1, blk.b_1, blk.W_2, blk.b_2);
Hn = R + G;                                                    % second residual
c = struct('H', H, 'U', U, 'ln1', ln1, 'att', att, 'F', F, 'R', R, 'D', D, 'ln2', ln2, ...
           'ffn', ffn, 'G', G);
end
