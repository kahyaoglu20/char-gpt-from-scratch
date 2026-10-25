function g = attention_backward(G_F, cache, W_QKV, W_O)
%ATTENTION_BACKWARD Matrix-calculus backward pass of causal attention (pages).
%   G_WO = Z'G_F,  G_Z = G_F W_O',  G_A = G_Z V',  G_V = A'G_Z,
%   G_S = A.*(G_A - rowsum(A.*G_A)),  G_Q = G_S K/sqrt(d),  G_K = G_S'Q/sqrt(d),
%   G_WQKV = U'[G_Q G_K G_V],  G_U = G_Q W_Q' + G_K W_K' + G_V W_V'.
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
[B, T, d] = size(cache.U);
A = cache.A;
G_WO = reshape(cache.Zb, B*T, d)' * reshape(G_F, B*T, d);   % sum_b Z_b' G_F,b    d x d
G_Z = pagemtimes(permute(G_F, [2 3 1]), W_O');      % G_F W_O'              T x d x B
G_A = pagemtimes(G_Z, 'none', cache.V, 'transpose');    % G_Z V'            T x T x B
G_V = pagemtimes(A, 'transpose', G_Z, 'none');      % A' G_Z                T x d x B
G_S = A .* (G_A - sum(A .* G_A, 2));                % row-wise softmax Jacobian
G_Q = pagemtimes(G_S, cache.K) / sqrt(d);           % G_S K / sqrt(d)       T x d x B
G_K = pagemtimes(G_S, 'transpose', cache.Q, 'none') / sqrt(d);   % G_S' Q / sqrt(d)
G_QKV = reshape(permute(cat(2, G_Q, G_K, G_V), [3 1 2]), B*T, 3*d);   % (BT) x 3d
G_WQKV = reshape(cache.U, B*T, d)' * G_QKV;         % [U'G_Q U'G_K U'G_V]   d x 3d
G_U = reshape(G_QKV * W_QKV', B, T, d);             % back to B x T x d
g = struct('W_O', G_WO, 'Z', G_Z, 'A', G_A, 'V', G_V, 'S', G_S, 'Q', G_Q, ...
           'K', G_K, 'W_QKV', G_WQKV, 'U', G_U);
end
