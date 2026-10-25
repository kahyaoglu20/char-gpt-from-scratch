function P = unpack_params(F, L)
%UNPACK_PARAMS Inverse of pack_params for an L-block model (works on dlarray fields).
names = {'W_QKV', 'W_O', 'W_1', 'b_1', 'W_2', 'b_2', 'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta'};
blocks = cell(1, L);
for l = 1:L
    blk = struct();
    for k = 1:numel(names)
        blk.(names{k}) = F.(sprintf('b%d_%s', l, names{k}));
    end
    blocks{l} = blk;
end
P = struct('W_E', F.W_E, 'W_P', F.W_P, 'blocks', [blocks{:}], 'lnf_gamma', F.lnf_gamma, ...
           'lnf_beta', F.lnf_beta, 'W_voc', F.W_voc);
end
