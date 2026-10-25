function G = cross_entropy_backward(Z, y)
%CROSS_ENTROPY_BACKWARD dL/dZ = (P - E) / R with E the one-hot targets;
%   every row sums to zero.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
R = size(Z, 1);
[~, G] = cross_entropy(Z, y);
idx = sub2ind(size(Z), (1:R)', y(:));
G(idx) = G(idx) - 1;
G = G / R;
end
