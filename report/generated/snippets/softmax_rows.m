function A = softmax_rows(S)
%SOFTMAX_ROWS Stable softmax across the key index (dimension 2) of every page.
m = max(S, [], 2);                                  % row maxima m_i
E = exp(S - m);                                     % entries <= 1: no overflow
A = E ./ sum(E, 2);                                 % every row sums to one
end
