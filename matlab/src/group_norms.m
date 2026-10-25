function n = group_norms(G)
%GROUP_NORMS Frobenius norms of the four parameter groups of a flat gradient struct:
%   [embeddings (W_E, W_P), attention (W_QKV, W_O), feed-forward (W_1, W_2), W_voc].
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
fn = fieldnames(G);
s = zeros(1, 4);
for k = 1:numel(fn)
    name = fn{k};
    q = sum(G.(name).^2, 'all');
    if any(strcmp(name, {'W_E', 'W_P'}))
        s(1) = s(1) + q;
    elseif endsWith(name, '_W_QKV') || endsWith(name, '_W_O')
        s(2) = s(2) + q;
    elseif endsWith(name, '_W_1') || endsWith(name, '_W_2')
        s(3) = s(3) + q;
    elseif strcmp(name, 'W_voc')
        s(4) = s(4) + q;
    end
end
n = sqrt(s);
end
