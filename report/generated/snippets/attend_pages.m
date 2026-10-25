function [S, A, Z] = attend_pages(Q, K, Vm, neg)
%ATTEND_PAGES Causal attention on T x d x B pages (one page per sample).
%   S = Q K'/sqrt(d) + M,  A = softmax_row(S),  Z = A Vm.
if nargin < 4
    neg = -1e9;
end
fwd_counter('inc', 'attention');
[T, d, ~] = size(Q);
S = pagemtimes(Q, 'none', K, 'transpose') / sqrt(d) + causal_mask(T, neg);   % T x T x B
A = softmax_rows(S);                                % T x T x B
Z = pagemtimes(A, Vm);                              % T x d x B
end
