function [G_WE, G_WP] = embed_backward(G_H, O, T_max)
%EMBED_BACKWARD dL/dW_E = sum_b O_b' D_b,  dL/dW_P(t,:) = sum_b D_b(t,:).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
[B, T, d] = size(G_H);
V = size(O, 3);
G_WE = reshape(O, B*T, V)' * reshape(G_H, B*T, d);  % (V x BT)(BT x d) = V x d
G_WP = zeros(T_max, d, 'like', G_H);
G_WP(1:T, :) = reshape(sum(G_H, 1), T, d);          % positions aggregate over samples
end
