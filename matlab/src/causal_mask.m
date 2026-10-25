function M = causal_mask(T, neg)
%CAUSAL_MASK M(i,j) = 0 for j <= i and -1e9 for j > i (T x T).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
if nargin < 2
    neg = -1e9;
end
M = triu(neg * ones(T), 1);
end
