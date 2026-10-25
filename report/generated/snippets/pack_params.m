function F = pack_params(P)
%PACK_PARAMS One-level struct of every tensor: P.blocks(l).name -> F.b<l>_name.
%   Adam and dlgradient work on this flat form; unpack_params inverts it.
F = struct('W_E', P.W_E, 'W_P', P.W_P);
for l = 1:numel(P.blocks)
    fn = fieldnames(P.blocks(l));
    for k = 1:numel(fn)
        F.(sprintf('b%d_%s', l, fn{k})) = P.blocks(l).(fn{k});
    end
end
F.lnf_gamma = P.lnf_gamma;
F.lnf_beta = P.lnf_beta;
F.W_voc = P.W_voc;
end
