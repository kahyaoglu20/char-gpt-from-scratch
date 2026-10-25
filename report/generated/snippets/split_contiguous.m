function [train, val, n_tr] = split_contiguous(x)
%SPLIT_CONTIGUOUS Chronological 90/10 split before any window is sampled.
%   n_tr = floor(0.9 N) (computed as floor(9N/10), exact for integer N).
N = numel(x);
n_tr = floor(9 * N / 10);
train = x(1:n_tr);                       % (c_1, ..., c_{n_tr})
val = x(n_tr + 1:end);                   % (c_{n_tr+1}, ..., c_N)
end
