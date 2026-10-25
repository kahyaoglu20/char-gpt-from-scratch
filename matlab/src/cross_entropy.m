function [L, P] = cross_entropy(Z, y)
%CROSS_ENTROPY Mean next-token negative log-likelihood with a stable log-sum-exp:
%   LSE_r = a_r + log sum_j exp(Z_rj - a_r),  a_r = max_j Z_rj,
%   L = -(1/R) sum_r (Z_{r,y_r} - LSE_r).   y holds 1-based labels, one per row of Z.
%   Vectorized gathering with sub2ind (no loop over tokens). Works on dlarray Z.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
R = size(Z, 1);
a = max(Z, [], 2);                                  % R x 1 row maxima: every exponent is <= 0
lse = a + log(sum(exp(Z - a), 2));                  % R x 1
idx = sub2ind(size(Z), (1:R)', y(:));               % linear index of Z(r, y_r)
L = mean(lse - Z(idx));
if nargout > 1
    P = exp(Z - lse);                               % softmax rows
end
end
