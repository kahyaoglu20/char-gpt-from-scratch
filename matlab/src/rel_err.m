function r = rel_err(a, b)
%REL_ERR The assignment's relative discrepancy |a-b| / max(1, |a|, |b|).
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
r = abs(a - b) / max([1, abs(a), abs(b)]);
end
