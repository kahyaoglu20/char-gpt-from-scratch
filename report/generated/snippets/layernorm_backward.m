function [G_X, G_gamma, G_beta] = layernorm_backward(G_Y, cache, gamma)
%LAYERNORM_BACKWARD Componentwise LayerNorm derivative with q = g .* gamma:
%   dL/dx_j = (1/s) [q_j - mean_k q_k - xhat_j mean_k(q_k xhat_k)].
d = size(G_Y, 3);
xhat = cache.xhat;
q = G_Y .* reshape(gamma, 1, 1, d);
G_X = (q - mean(q, 3) - xhat .* mean(q .* xhat, 3)) ./ cache.s;
G_gamma = reshape(sum(sum(G_Y .* xhat, 1), 2), 1, d);   % sum over all tokens
G_beta = reshape(sum(sum(G_Y, 1), 2), 1, d);
end
