function [F, state, delta] = adam_step(F, G, state, lr, beta1, beta2, eps_a)
%ADAM_STEP One Adam update of every field of the flat parameter struct F:
%   m_t = b1 m + (1-b1) g,  v_t = b2 v + (1-b2) g.*g,
%   m^ = m_t / (1 - b1^t),  v^ = v_t / (1 - b2^t),  theta = theta - lr m^ ./ (sqrt(v^) + eps).
%   No clipping, no weight decay. delta returns the applied steps.
state.t = state.t + 1;
c1 = 1 - beta1^state.t;                             % bias corrections (moments start at zero)
c2 = 1 - beta2^state.t;
fn = fieldnames(F);
delta = struct();
for k = 1:numel(fn)
    n = fn{k};
    g = G.(n);
    state.m.(n) = beta1 * state.m.(n) + (1 - beta1) * g;
    state.v.(n) = beta2 * state.v.(n) + (1 - beta2) * (g .* g);
    step = -lr * (state.m.(n) / c1) ./ (sqrt(state.v.(n) / c2) + eps_a);
    F.(n) = F.(n) + step;
    delta.(n) = step;
end
end
