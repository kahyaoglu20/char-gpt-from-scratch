function [F, cache] = attention_forward(U, W_QKV, W_O, neg)
%ATTENTION_FORWARD Single-head causal self-attention of a B x T x d tensor.
%   Dimension trace: (BT) x d -> (BT) x 3d -> B x T x 3d -> T x d x B
%                    -> T x T x B -> B x T x d.
if nargin < 4
    neg = -1e9;
end
[B, T, d] = size(U);
QKV = reshape(U, B*T, d) * W_QKV;                   % (BT x d)(d x 3d) = (BT) x 3d
QKV = reshape(QKV, B, T, 3*d);                      % B x T x 3d
Q = permute(QKV(:, :, 1:d), [2 3 1]);               % T x d x B queries
K = permute(QKV(:, :, d+1:2*d), [2 3 1]);           % T x d x B keys
Vm = permute(QKV(:, :, 2*d+1:3*d), [2 3 1]);        % T x d x B values (not vocab size)
[S, A, Z] = attend_pages(Q, K, Vm, neg);            % T x T x B, T x T x B, T x d x B
Zb = permute(Z, [3 1 2]);                           % B x T x d
F = reshape(reshape(Zb, B*T, d) * W_O, B, T, d);    % (BT x d)(d x d) -> B x T x d
cache = struct('U', U, 'Q', Q, 'K', K, 'V', Vm, 'S', S, 'A', A, 'Z', Z, 'Zb', Zb);
end
