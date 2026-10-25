function P = init_params(V, T, d, L, sigma)
%INIT_PARAMS Every trainable tensor of the L-block character GPT.
%   Weights ~ N(0, sigma^2), biases = 0, LayerNorm gains = 1, shifts = 0.
%   Right-multiplication convention Y = X W: weights are d_in x d_out.
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
P.W_E = sigma * randn(V, d);             % token embeddings      V x d
P.W_P = sigma * randn(T, d);             % position embeddings   T x d
for l = 1:L
    P.blocks(l).W_QKV = sigma * randn(d, 3*d);   % [W_Q W_K W_V]   d x 3d
    P.blocks(l).W_O = sigma * randn(d, d);       % output proj.    d x d
    P.blocks(l).W_1 = sigma * randn(d, 4*d);     % FFN expand      d x 4d
    P.blocks(l).b_1 = zeros(1, 4*d);             %                 1 x 4d
    P.blocks(l).W_2 = sigma * randn(4*d, d);     % FFN contract    4d x d
    P.blocks(l).b_2 = zeros(1, d);               %                 1 x d
    P.blocks(l).ln1_gamma = ones(1, d);          % LN1 gain/shift  1 x d each
    P.blocks(l).ln1_beta = zeros(1, d);
    P.blocks(l).ln2_gamma = ones(1, d);          % LN2 gain/shift  1 x d each
    P.blocks(l).ln2_beta = zeros(1, d);
end
P.lnf_gamma = ones(1, d);                % final LN gain/shift   1 x d each
P.lnf_beta = zeros(1, d);
P.W_voc = sigma * randn(d, V);           % vocabulary projection d x V (untied)
end
