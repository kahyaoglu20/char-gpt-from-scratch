function vocab = build_vocab(text)
%BUILD_VOCAB Sorted unique characters c_1 < ... < c_V (1 x V char row).
%   The bijection f maps vocab(k) -> k, k = 1..V (MATLAB indices are 1-based).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
vocab = unique(text);                    % unique() returns the characters sorted
end
