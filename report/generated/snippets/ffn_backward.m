function g = ffn_backward(G_G, cache, W_1, W_2)
%FFN_BACKWARD Hand-derived gradients of the feed-forward network:
%   G_W2 = g(J)' G_G,  G_b2 = 1' G_G,  G_g = G_G W2',  G_J = G_g .* g'(J),
%   G_W1 = D' G_J,  G_b1 = 1' G_J,  G_D = G_J W1'   (sums over all BT rows).
[B, T, d] = size(G_G);
G2 = reshape(G_G, B*T, d);                          % (BT) x d
g.W_2 = cache.Ga' * G2;                             % 4d x d
g.b_2 = sum(G2, 1);                                 % 1 x d
g.Ga = G2 * W_2';                                   % (BT) x 4d
g.J = g.Ga .* gelu_grad(cache.J);                   % elementwise chain rule through g
g.W_1 = cache.D2' * g.J;                            % d x 4d
g.b_1 = sum(g.J, 1);                                % 1 x 4d
g.D = reshape(g.J * W_1', B, T, d);                 % back to B x T x d
end
