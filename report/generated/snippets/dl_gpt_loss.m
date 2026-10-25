function [J, G] = dl_gpt_loss(F, X, y, L, eps_ln)
%DL_GPT_LOSS Traced training objective for dlfeval: the flat parameter struct F holds
%   dlarray copies of every tensor, so dlgradient differentiates the whole GPT
%   (embedding gathers, both LayerNorms, masked softmax, GELU, residual sums, head).
%   [J, G] = dlfeval(@dl_gpt_loss, F, X, y, L, eps_ln)
P = unpack_params(F, L);
Z = gpt_forward(X, P, eps_ln);
J = cross_entropy(Z, y);
G = dlgradient(J, F);
end
