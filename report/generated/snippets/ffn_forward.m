function [G, cache] = ffn_forward(D, W_1, b_1, W_2, b_2)
%FFN_FORWARD Feed-forward network of every token row: G = g(D W1 + 1 b1) W2 + 1 b2.
%   D is B x T x d. No token interacts with another, so the B*T rows are
%   flattened into one (BT) x d matrix; the 1 x 4d and 1 x d bias rows are
%   added to every row by implicit expansion.
[B, T, d] = size(D);
D2 = reshape(D, B*T, d);                            % (BT) x d
J = D2 * W_1 + b_1;                                 % (BT x d)(d x 4d) + 1 x 4d
Ga = gelu_tanh(J);                                  % (BT) x 4d
G = reshape(Ga * W_2 + b_2, B, T, d);               % (BT x 4d)(4d x d) + 1 x d -> B x T x d
cache = struct('D2', D2, 'J', J, 'Ga', Ga);
end
