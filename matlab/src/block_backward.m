function [G_H, g] = block_backward(G_Hn, c, blk)
%BLOCK_BACKWARD Backward pass of one block. Each residual sum passes the upstream
%   gradient along the identity path unchanged and adds the branch's share:
%   dL/dX = G_Y + (dF/dX)' G_Y.  The fields of g follow the order of init_params.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
gf = ffn_backward(G_Hn, c.ffn, blk.W_1, blk.W_2);
[G_R_ln, g_ln2, b_ln2] = layernorm_backward(gf.D, c.ln2, blk.ln2_gamma);
G_R = G_Hn + G_R_ln;                                           % H' = R + G
ga = attention_backward(G_R, c.att, blk.W_QKV, blk.W_O);      % Assignment 1, unchanged
[G_H_ln, g_ln1, b_ln1] = layernorm_backward(ga.U, c.ln1, blk.ln1_gamma);
G_H = G_R + G_H_ln;                                            % R = H + F
g = struct('W_QKV', ga.W_QKV, 'W_O', ga.W_O, 'W_1', gf.W_1, 'b_1', gf.b_1, 'W_2', gf.W_2, ...
           'b_2', gf.b_2, 'ln1_gamma', g_ln1, 'ln1_beta', b_ln1, 'ln2_gamma', g_ln2, 'ln2_beta', b_ln2);
end
