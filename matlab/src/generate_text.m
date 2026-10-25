function txt = generate_text(P, prompt, vocab, n_new, T, eps_ln, stream, temperature, greedy)
%GENERATE_TEXT Autoregressive sampling from a prompt, one character at a time.
%   Keep at most the last T indices (positions re-indexed 1..n), run the model,
%   use only the last row of logits, softmax with temperature, then draw one index
%   by inverse CDF (or take the argmax when greedy is true).
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
if nargin < 8, temperature = 1; end
if nargin < 9, greedy = false; end
ids = encode_text(prompt, vocab);
for k = 1:n_new
    ctx = ids(max(1, end - T + 1):end);             % the newest T tokens
    Z = gpt_forward(ctx, P, eps_ln);                % B = 1, so row t of Z is position t
    z = Z(end, :) / temperature;                    % only the last row predicts the next character
    p = exp(z - max(z));
    p = p / sum(p);
    if greedy
        [~, nxt] = max(p);
    else
        nxt = find(cumsum(p) >= rand(stream), 1);
        if isempty(nxt), nxt = numel(p); end        % guards against round-off in the last bin
    end
    ids(end + 1) = nxt; %#ok<AGROW>
end
txt = decode_idx(ids, vocab);
end
