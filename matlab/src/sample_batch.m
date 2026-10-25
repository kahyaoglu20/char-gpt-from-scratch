function [X, Y, r] = sample_batch(data, B, T, stream)
%SAMPLE_BATCH B shifted windows wholly inside one split, drawn from a given RandStream
%   (training, validation and sampling use separate streams, so the evaluation
%   schedule never changes which training windows are drawn).
%   r_b ~ Uniform{1, ..., M-T},  X(b,t) = x(r_b+t-1),  Y(b,t) = x(r_b+t).
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
M = numel(data);
if M <= T
    error('sample_batch:short', 'split of length %d too short for T = %d', M, T);
end
r = randi(stream, [1, M - T], B, 1);                % B x 1 legal starts
idx = r + (0:T-1);                                  % B x T positions (implicit expansion)
X = reshape(data(idx), B, T);
Y = reshape(data(idx + 1), B, T);                   % shifted one step: the next characters
end
