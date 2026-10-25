function [X, Y, r] = get_batch(data, B, T)
%GET_BATCH B shifted windows wholly inside one split (1-based indices).
%   r_b ~ Uniform{1, ..., M-T},  X(b,t) = x(r_b+t-1),  Y(b,t) = x(r_b+t).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
M = numel(data);
if M <= T
    error('get_batch:short', 'split of length %d too short for T = %d', M, T);
end
r = randi([1, M - T], B, 1);             % B x 1 legal starts
idx = r + (0:T-1);                       % B x T positions (implicit expansion)
X = data(idx);                           % B x T inputs
Y = data(idx + 1);                       % B x T targets, shifted one step left
X = reshape(X, B, T);                    % keep B x T even when B = 1 or T = 1
Y = reshape(Y, B, T);
end
