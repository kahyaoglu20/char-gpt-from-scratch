function [H, O] = embed_forward(X, W_E, W_P)
%EMBED_FORWARD H0_b = O_b W_E + W_P(1:T,:), stacked into a B x T x d tensor.
%   Works for numeric arrays and for traced dlarray tables (dlgradient).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
fwd_counter('inc', 'embed');
[B, T] = size(X);
[V, d] = size(W_E);
O = one_hot(X, V, underlying(W_E));                 % B x T x V constant selector
E = reshape(reshape(O, B*T, V) * W_E, B, T, d);     % (BT x V)(V x d) -> B x T x d
H = E + reshape(W_P(1:T, :), 1, T, d);              % position rows broadcast over b
end

function z = underlying(w)
% zero scalar of the numeric class underneath w (double or single)
if isa(w, 'dlarray')
    w = extractdata(w);
end
z = zeros(1, 'like', w);
end
