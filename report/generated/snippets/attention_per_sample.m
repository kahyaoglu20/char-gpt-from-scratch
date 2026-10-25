function F = attention_per_sample(U, W_QKV, W_O, neg)
%ATTENTION_PER_SAMPLE Reference: one sample at a time with plain 2-D matrices.
if nargin < 4
    neg = -1e9;
end
[B, T, d] = size(U);
M = causal_mask(T, neg);
W_Q = W_QKV(:, 1:d);  W_K = W_QKV(:, d+1:2*d);  W_V = W_QKV(:, 2*d+1:3*d);
F = zeros(B, T, d, 'like', U);
for b = 1:B
    Ub = reshape(U(b, :, :), T, d);                 % T x d
    Q = Ub * W_Q;  K = Ub * W_K;  V = Ub * W_V;     % T x d each
    S = Q * K' / sqrt(d) + M;                       % T x T
    E = exp(S - max(S, [], 2));
    A = E ./ sum(E, 2);                             % rows sum to one
    F(b, :, :) = reshape(A * V * W_O, 1, T, d);
end
end
