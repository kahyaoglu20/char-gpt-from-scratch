function [Y, cache] = layernorm_forward(H, gamma, beta, eps)
%LAYERNORM_FORWARD Feature-wise LayerNorm of every token of a B x T x d tensor.
%   mu = (1/d) H 1_d,  C = H - mu 1_d',  v = (1/d)(C.*C) 1_d,
%   LN(H) = C ./ sqrt(v + eps) .* gamma + beta   (gamma, beta are 1 x d rows).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
fwd_counter('inc', 'layernorm');
d = size(H, 3);
mu = mean(H, 3);                                    % B x T x 1 token means
C = H - mu;                                         % B x T x d centred features
v = mean(C .* C, 3);                                % B x T x 1 population variances
s = sqrt(v + eps);                                  % eps inside the square root
xhat = C ./ s;                                      % standardized rows
Y = xhat .* reshape(gamma, 1, 1, d) + reshape(beta, 1, 1, d);
cache = struct('mu', mu, 'v', v, 's', s, 'xhat', xhat);
end
