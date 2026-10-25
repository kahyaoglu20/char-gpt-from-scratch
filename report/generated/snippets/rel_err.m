function r = rel_err(a, b)
%REL_ERR The assignment's relative discrepancy |a-b| / max(1, |a|, |b|).
r = abs(a - b) / max([1, abs(a), abs(b)]);
end
