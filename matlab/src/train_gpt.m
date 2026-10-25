function [F, rec] = train_gpt(F, train, val, cfg, streams, vocab)
%TRAIN_GPT The optimization loop: fresh random training windows, cross-entropy,
%   gradients (dlgradient or the hand-derived chain rule, cfg.grad), one Adam step.
%   At step 1 and every cfg.eval_every steps a fresh batch of min(8, B) windows is
%   drawn from the validation part and evaluated AFTER the update, without any
%   gradient or optimizer call; the Adam state is checked to be unchanged.
%
%   S_epoch = floor(N_train / (B T));  S_total = cfg.epochs S_epoch, capped by cfg.s_max > 0.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
s_epoch = floor(numel(train) / (cfg.B * cfg.T));
s_total = cfg.epochs * s_epoch;
if cfg.s_max > 0
    s_total = min(s_total, cfg.s_max);
end
state = adam_init(F);
train_curve = zeros(s_total, 1);
evals = zeros(0, 3);
norms = zeros(0, 5);
previews = struct('step', {}, 'text', {});
val_ok = true;
B_val = min(8, cfg.B);
t0 = tic;
for s = 1:s_total
    [X, Y] = sample_batch(train, cfg.B, cfg.T, streams.train);
    [Ls, G] = loss_and_grad(F, X, Y(:), cfg.L, cfg.eps, cfg.grad);
    gvals = struct2cell(G);
    if ~all(cellfun(@(x) all(isfinite(x), 'all'), gvals))
        error('train_gpt:nonfinite', 'non-finite gradient at step %d', s);
    end
    train_curve(s) = Ls;                            % training loss, measured before the update
    [F, state] = adam_step(F, G, state, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps_adam);
    if s == 1 || mod(s, cfg.eval_every) == 0
        before = adam_fingerprint(state);
        [Xv, Yv, rv] = sample_batch(val, B_val, cfg.T, streams.val);   % held-out part only
        Lv = cross_entropy(gpt_forward(Xv, unpack_params(F, cfg.L), cfg.eps), Yv(:));
        val_ok = val_ok && isequal(before, adam_fingerprint(state)) && min(rv) >= 1 && max(rv) <= numel(val) - cfg.T;
        evals(end + 1, :) = [s, Ls, Lv]; %#ok<AGROW>
        norms(end + 1, :) = [s, group_norms(G)]; %#ok<AGROW>
        if s == 1 || mod(s, 20 * cfg.eval_every) == 0 || s == s_total
            fprintf('step %6d/%d  epoch %5.2f  train %.4f  val %.4f  (%.1f steps/s)\n', ...
                    s, s_total, s / s_epoch, Ls, Lv, s / toc(t0));
        end
    end
    if cfg.preview_every > 0 && (mod(s, cfg.preview_every) == 0 || s == s_total)
        txt = generate_text(unpack_params(F, cfg.L), cfg.prompt, vocab, cfg.preview_len, cfg.T, cfg.eps, streams.preview);
        previews(end + 1) = struct('step', s, 'text', txt); %#ok<AGROW>
        fprintf('   preview @ %d: %s\n', s, strrep(txt, newline, '\n'));
    end
end
rec = struct('s_epoch', s_epoch, 's_total', s_total, 'train_curve', train_curve, 'evals', evals, ...
             'norms', norms, 'previews', previews, 'val_ok', val_ok, 'seconds', toc(t0), 'adam_steps', state.t);
end

function fp = adam_fingerprint(state)
% a cheap digest of the optimizer state: step count and the sums of both moments
m = struct2cell(state.m);
v = struct2cell(state.v);
fp = [state.t, sum(cellfun(@(x) sum(abs(x), 'all'), m)), sum(cellfun(@(x) sum(x, 'all'), v))];
end
