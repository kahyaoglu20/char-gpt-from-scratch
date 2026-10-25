function O = one_hot(X, V, like)
%ONE_HOT O(b,t,:) = e_{X(b,t)}^T : the B x T x V one-hot selection tensor.
%   The optional third argument fixes the class (double or single).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
if nargin < 3
    like = 0;
end
[B, T] = size(X);
O = zeros(B, T, V, 'like', like);
[b, t] = ndgrid(1:B, 1:T);
O(sub2ind([B, T, V], b(:), t(:), X(:))) = 1;    % exactly one unit entry per (b, t)
end
