function [L, G] = loss_and_grad(F, X, y, n_blocks, eps_ln, mode)
%LOSS_AND_GRAD Training loss and the gradient of every tensor (flat struct G).
%   mode 'dlgradient': automatic differentiation with the Deep Learning Toolbox.
%   mode 'manual':     the hand-derived chain rule (gpt_backward), no toolbox needed.
if strcmp(mode, 'dlgradient')
    Fd = structfun(@dlarray, F, 'UniformOutput', false);
    [Ld, Gd] = dlfeval(@dl_gpt_loss, Fd, X, y, n_blocks, eps_ln);
    L = double(extractdata(Ld));
    G = structfun(@extractdata, Gd, 'UniformOutput', false);
else
    P = unpack_params(F, n_blocks);
    [Z, c] = gpt_forward(X, P, eps_ln);
    L = cross_entropy(Z, y);
    G = pack_params(gpt_backward(cross_entropy_backward(Z, y), c, P));
end
end
