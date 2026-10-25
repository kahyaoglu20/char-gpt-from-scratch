%% run_assignment2.m -- MAT 451E Assignment 2 (character-level GPT: training and generation), MATLAB
% Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741
%
% Runs every question (Q1-Q6), Benchmark B, Cases B1-B5 and the minimum acceptance
% tests, trains the GPT on Tiny Shakespeare for 50 epochs and generates text.
% The check identifiers are the same as in python/run_assignment2.py.
%
%   >> cd matlab, run_assignment2
%   $  matlab -batch "run('matlab/run_assignment2.m')"          (from the submission root)
%
% Optional environment variables (set before starting MATLAB, or with setenv):
%   A2_STEPS=500     cap the training at 500 steps (S_max) for a quick run
%   A2_GRAD=manual   train with the hand-derived chain rule instead of dlgradient
%
% Outputs (relative paths): results/matlab/{log.txt, results.json, checks.csv,
% training_log.csv, samples.txt, model_final.mat, figures/loss_curves_matlab.png}.
% Helpers live in matlab/src (Assignment 1 functions reused unchanged, plus the new ones).

clearvars; close all;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(here, 'src'));
data_file = fullfile(root, 'data', 'tiny_shakespeare.txt');
a1_dir = fullfile(root, 'data', 'assignment1');
out_dir = fullfile(root, 'results', 'matlab');
fig_dir = fullfile(out_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
log_file = fullfile(out_dir, 'log.txt');
if exist(log_file, 'file'), delete(log_file); end
diary(log_file);
record_check('reset');
fwd_counter('reset');
t_start = tic;

SEED = 451;
CFG = struct('B', 32, 'T', 64, 'd', 64, 'L', 2, 'd_ff', 256, 'sigma', 0.02, 'eps', 1e-5, ...
             'mask_value', -1e9, 'lr', 1e-3, 'beta1', 0.9, 'beta2', 0.999, 'eps_adam', 1e-8, ...
             'epochs', 50, 's_max', 0, 'eval_every', 50, 'preview_every', 2450, 'preview_len', 80, ...
             'sample_len', 200, 'prompt', 'ROMEO: ', 'grad', 'dlgradient', 'seed', SEED, ...
             'fixed_eval_windows', 512, 'index_base', 1);
if ~isempty(getenv('A2_STEPS')), CFG.s_max = str2double(getenv('A2_STEPS')); end
has_dl = ~isempty(ver('nnet')) && license('test', 'Neural_Network_Toolbox');
if ~has_dl || strcmpi(getenv('A2_GRAD'), 'manual'), CFG.grad = 'manual'; end
H_VALUES = [1e-3 1e-4 1e-5 1e-6];
B = CFG.B; T = CFG.T; d = CFG.d; L = CFG.L; eps_ln = CFG.eps;
R = struct();

%% Environment
print_section('Environment');
if has_dl, tb = ver('nnet'); tbv = tb.Version; else, tbv = 'not installed'; end
R.env = struct('matlab', version, 'release', version('-release'), 'deep_learning_toolbox', tbv, ...
               'computer', computer, 'float', 'IEEE-754 double', 'gradient', CFG.grad);
fprintf('MATLAB %s, Deep Learning Toolbox %s, %s; training gradients: %s\n', version, tbv, computer, CFG.grad);

%% Data, vocabulary and the Assignment 1 handoff
print_section('Data, vocabulary and the Assignment 1 handoff');
[corpus, raw] = load_corpus(data_file);
vocab = build_vocab(corpus);
x = encode_text(corpus, vocab);
N = numel(x);  V = numel(vocab);
[train, val, n_tr] = split_contiguous(x);
sha = sha256_bytes(raw);
fprintf('N = %d, V = %d, N_tr = %d, N_va = %d, SHA-256 %s\n', N, V, numel(train), numel(val), sha);
S1 = load(fullfile(a1_dir, 'params_init.mat'));     % P, vocab and CFG saved by Assignment 1 (MATLAB)
record_check('D.1', 'Q6', 'vocabulary differs from Assignment 1 (characters)', double(~isequal(S1.vocab, vocab)), 0, '0', '==');
P0 = S1.P;
record_check('D.2', 'Q6', 'P(loaded) - (128V + 103680)', count_params(P0) - (128 * V + 103680), 0, '0', '==');
rng(SEED, 'twister');
Pf = init_params(V, T, d, L, CFG.sigma);
record_check('D.3', 'Q6', 'saved Assignment 1 weights vs re-running its init (rng 451)', max_struct_diff(pack_params(P0), pack_params(Pf)), 0, '0', '==');
Xf = encode_text('ROMEO: But soft, what light', vocab);
record_check('D.4', 'Q6', 'fixed-input logits: saved vs re-created weights', ...
             max(abs(gpt_forward(Xf, P0, eps_ln) - gpt_forward(Xf, Pf, eps_ln)), [], 'all'), 0, '0', '==');
R.data = struct('file', 'tiny_shakespeare.txt', 'sha256', sha, 'N', N, 'V', V, 'N_tr', numel(train), 'N_va', numel(val));

%% Q1: residual connections and the feed-forward network
print_section('Q1  Residual connections and the feed-forward network');
Drow = [0.5 -1.0];
W1s = [0.4 -0.3 0.2 0.1 -0.5 0.6 0.0 0.3; 0.1 0.2 -0.4 0.5 0.3 -0.1 0.7 -0.2];
b1s = [0.0 0.1 -0.1 0.2 0.0 -0.2 0.1 0.0];
W2s = [0.3 -0.2; 0.1 0.4; -0.5 0.2; 0.2 0.1; 0.0 -0.3; 0.4 0.0; -0.1 0.5; 0.2 0.2];
b2s = [0.05 -0.05];
[Gs, cs] = ffn_forward(reshape(Drow, 1, 1, 2), W1s, b1s, W2s, b2s);
J_loop = zeros(1, 8);  G_loop = zeros(1, 2);
for k = 1:8, J_loop(k) = Drow(1) * W1s(1, k) + Drow(2) * W1s(2, k) + b1s(k); end
g_loop = 0.5 * J_loop .* (1 + tanh(sqrt(2/pi) * (J_loop + 0.044715 * J_loop.^3)));
for j = 1:2, G_loop(j) = sum(g_loop .* W2s(:, j)') + b2s(j); end
show('J = D W1 + b1', J_loop, '%8.4f');
show('G = g(J) W2 + b2', G_loop, '%9.5f');
record_check('Q1.1', 'Q1', 'd=2 example: vectorized FFN vs the elementwise sums', ...
             max([max(abs(cs.J - J_loop)), max(abs(reshape(Gs, 1, 2) - G_loop))]), 1e-15);
z3 = [-1 0 1];
exact = z3 .* 0.5 .* (1 + erf(z3 / sqrt(2)));
approx = gelu_tanh(z3);
fprintf('GELU at z = -1, 0, 1: approximation %s, exact %s\n', mat2str(approx, 10), mat2str(exact, 10));
record_check('Q1.2', 'Q1', 'max |tanh-GELU - z Phi(z)| at z = -1, 0, 1', max(abs(approx - exact)), 1e-3, '~1.6e-4');
zz = linspace(-6, 6, 2401);
if has_dl
    [~, gz] = dlfeval(@gelu_sum, dlarray(zz));
    e_gp = max(abs(extractdata(gz) - gelu_grad(zz)));
    record_check('Q1.3', 'Q1', 'g''(z) formula vs dlgradient of gelu_tanh (2401 points)', e_gp, 1e-14);
else
    e_gp = max(abs((gelu_tanh(zz + 1e-5) - gelu_tanh(zz - 1e-5)) / 2e-5 - gelu_grad(zz)));
    record_check('Q1.3', 'Q1', 'g''(z) formula vs centred differences (2401 points)', e_gp, 1e-9);
end
rng(SEED, 'twister');
Tq = 4; dq = 3;
Pm = small_params(5, Tq, dq, 1);
Hq = randn(1, Tq, dq);
nq = Tq * dq;
JR = zeros(nq);  JF = zeros(nq);
for j = 1:nq
    e = zeros(1, Tq, dq);  e(j) = 1e-6;
    [Rp, Fp] = residual_branch(Hq + e, Pm.blocks(1), eps_ln);
    [Rm, Fm] = residual_branch(Hq - e, Pm.blocks(1), eps_ln);
    JR(:, j) = (Rp(:) - Rm(:)) / 2e-6;
    JF(:, j) = (Fp(:) - Fm(:)) / 2e-6;
end
record_check('Q1.4', 'Q1', 'residual Jacobian: max |dR/dH - J_F - I|', max(abs(JR - JF - eye(nq)), [], 'all'), 1e-8);
blockF = zeros(Tq);  blockG = zeros(Tq);
for i = 1:Tq
    for j = 1:Tq
        rows = i + (0:dq-1) * Tq;  cols = j + (0:dq-1) * Tq;     % entry (1, t, f) has linear index t + (f-1) T
        blockF(i, j) = max(abs(JF(rows, cols)), [], 'all');
    end
end
Dq = randn(1, Tq, dq);
G0 = ffn_forward(Dq, Pm.blocks(1).W_1, Pm.blocks(1).b_1, Pm.blocks(1).W_2, Pm.blocks(1).b_2);
for j = 1:Tq
    Dp = Dq;  Dp(1, j, :) = Dp(1, j, :) + 0.5;
    Gp = ffn_forward(Dp, Pm.blocks(1).W_1, Pm.blocks(1).b_1, Pm.blocks(1).W_2, Pm.blocks(1).b_2);
    blockG(:, j) = max(abs(Gp - G0), [], 3)';
end
show('attention: max |dF_i/dH_j|', blockF, '%9.2e');
show('feed-forward: max |change of G_i| after changing token j', blockG, '%9.2e');
record_check('Q1.5', 'Q1', 'FFN mixes features only: change of other tokens'' outputs', max(blockG(~eye(Tq)), [], 'all'), 0, '0', '==');
record_check('Q1.6', 'Q1', 'attention mixes tokens causally: max |dF_i/dH_j| for j > i', max(blockF(triu(true(Tq), 1))), 1e-8);
rng(SEED, 'twister');
Pm = small_params(7, 4, 5, 1);
Dn = randn(3, 4, 5);  Rg = randn(3, 4, 5);
blk = Pm.blocks(1);
[~, cq] = ffn_forward(Dn, blk.W_1, blk.b_1, blk.W_2, blk.b_2);
gm = ffn_backward(Rg, cq, blk.W_1, blk.W_2);
if has_dl
    [~, gW1, gb1, gW2, gb2, gD] = dlfeval(@ffn_objective, dlarray(Dn), dlarray(blk.W_1), dlarray(blk.b_1), ...
                                          dlarray(blk.W_2), dlarray(blk.b_2), Rg);
    e_ffn = max([max(abs(extractdata(gW1) - gm.W_1), [], 'all'), max(abs(extractdata(gb1) - gm.b_1), [], 'all'), ...
                 max(abs(extractdata(gW2) - gm.W_2), [], 'all'), max(abs(extractdata(gb2) - gm.b_2), [], 'all'), ...
                 max(abs(extractdata(gD) - gm.D), [], 'all')]);
    record_check('Q1.7', 'Q1', 'feed-forward backward: every term vs dlgradient', e_ffn, 1e-13);
else
    Jfd = @(W) sum(ffn_forward(Dn, W, blk.b_1, blk.W_2, blk.b_2) .* Rg, 'all');
    E = zeros(size(blk.W_1));  E(2, 3) = 1;
    e_ffn = rel_err((Jfd(blk.W_1 + 1e-5 * E) - Jfd(blk.W_1 - 1e-5 * E)) / 2e-5, gm.W_1(2, 3));
    record_check('Q1.7', 'Q1', 'feed-forward backward vs centred difference (W_1 entry)', e_ffn, 1e-9);
end
R.Q1 = struct('J', J_loop, 'gJ', g_loop, 'G', G_loop, 'gelu_approx', approx, 'gelu_exact', exact, ...
              'attention_blocks', blockF, 'ffn_blocks', blockG);

%% Case B2: random feed-forward and residual
print_section('Case B2  Random feed-forward and residual (seed 451, B=2, T=5, d=4, d_ff=16)');
rng(SEED, 'twister');
Hb2 = 0.02 * randn(2, 5, 4);  W1b = 0.02 * randn(4, 16);  W2b = 0.02 * randn(16, 4);
b1b = zeros(1, 16);  b2b = zeros(1, 4);
Db2 = layernorm_forward(Hb2, ones(1, 4), zeros(1, 4), eps_ln);
[Gb2, cb2] = ffn_forward(Db2, W1b, b1b, W2b, b2b);
Hn2 = Hb2 + Gb2;
fprintf('per sample: J %s, G %s, H_next %s\n', mat2str([5 16]), mat2str(size(Gb2, [2 3])), mat2str(size(Hn2, [2 3])));
record_check('B2.1', 'Q1', 'shape mismatches (J 5x16, G 5x4, H_next 5x4)', double(~isequal(size(cb2.J), [10 16]) || ~isequal(size(Hn2), [2 5 4])), 0, '0', '==');
record_check('B2.2', 'Q1', 'max |H_next - (H + G)|', max(abs(Hn2 - (Hb2 + Gb2)), [], 'all'), 0, '0', '==');
gb2 = ffn_backward(2 * Hn2, cb2, W1b, W2b);         % J = sum(H_next.^2), dJ/dH_next = 2 H_next
Jb2 = @(W1x, W2x) sum((Hb2 + ffn_forward(Db2, W1x, b1b, W2x, b2b)).^2, 'all');
worst = zeros(size(H_VALUES));  worst_s = zeros(size(H_VALUES));
for rep = 1:5
    i1 = randi(numel(W1b));  i2 = randi(numel(W2b));
    for k = 1:numel(H_VALUES)
        h = H_VALUES(k);
        E1 = zeros(size(W1b));  E1(i1) = 1;  E2 = zeros(size(W2b));  E2(i2) = 1;
        fd1 = (Jb2(W1b + h * E1, W2b) - Jb2(W1b - h * E1, W2b)) / (2 * h);
        fd2 = (Jb2(W1b, W2b + h * E2) - Jb2(W1b, W2b - h * E2)) / (2 * h);
        worst(k) = max([worst(k), rel_err(fd1, gb2.W_1(i1)), rel_err(fd2, gb2.W_2(i2))]);
        worst_s(k) = max([worst_s(k), abs(fd1 - gb2.W_1(i1)) / max(abs([fd1, gb2.W_1(i1)])), ...
                          abs(fd2 - gb2.W_2(i2)) / max(abs([fd2, gb2.W_2(i2)]))]);
    end
end
fprintf('FD vs hand-derived on 10 entries: %s (assignment measure), %s (scale-free)\n', mat2str(worst, 3), mat2str(worst_s, 3));
record_check('B2.3', 'Q1', 'FD vs gradient on 10 entries of W1, W2 (assignment measure, h=1e-5)', worst(3), 1e-9);
record_check('B2.4', 'Q1', 'FD vs gradient, scale-free relative error at h=1e-5', worst_s(3), 1e-5);
R.B2 = struct('worst_rel', worst, 'worst_strict', worst_s);

%% Q2: vocabulary projection and cross-entropy
print_section('Q2  Vocabulary projection and cross-entropy');
Zx = [2 0 -1; 0 1 0];  yx = [1; 2];
[Lx, Px] = cross_entropy(Zx, yx);
ref = 0.5 * (log(exp(2) + 1 + exp(-1)) - 2 + log(2 + exp(1)) - 1);
fprintf('L = %.15f, closed form %.15f\n', Lx, ref);
record_check('Q2.1', 'Q2', '2x3 example: |L - closed form|', abs(Lx - ref), 1e-15);
Gx = cross_entropy_backward(Zx, yx);
show('dL/dZ = (P - E)/2', Gx, '%10.6f');
record_check('Q2.2', 'Q2', 'dL/dZ: max |row sum| and |(P-E)/2 - formula|', ...
             max([max(abs(sum(Gx, 2))), max(abs(Gx - (Px - [1 0 0; 0 1 0]) / 2), [], 'all')]), 1e-16);
record_check('Q2.3', 'Q2', 'row shift by +-1000: |L(Z + c) - L(Z)| (stable LSE)', abs(cross_entropy(Zx + [1000; -1000], yx) - ref), 1e-12);
lab = [1 2 3; 4 5 6];
fprintf('labels [1 2 3; 4 5 6]: MATLAB column-major lab(:)'' = %s; row-major (Python) = %s\n', mat2str(lab(:)'), mat2str(reshape(lab', 1, [])));
rng(SEED, 'twister');
Hd = randn(2, 3, 4);  Wv = randn(4, 7);  labels = randi(7, 2, 3);
L_col = cross_entropy(reshape(Hd, 6, 4) * Wv, labels(:));
L_row = cross_entropy(reshape(permute(Hd, [2 1 3]), 6, 4) * Wv, reshape(labels', [], 1));
L_mix = cross_entropy(reshape(Hd, 6, 4) * Wv, reshape(labels', [], 1));
perm = randperm(6);
Zp = reshape(Hd, 6, 4) * Wv;  yl = labels(:);
L_perm = cross_entropy(Zp(perm, :), yl(perm));
record_check('Q2.4', 'Q2', 'same flattening rule (row- or column-major) or a joint permutation: max |delta L|', max(abs([L_col - L_row, L_col - L_perm])), 1e-14);
record_check('Q2.5', 'Q2', 'mismatched flattening changes the loss: |L_mix - L|', abs(L_mix - L_col), 1e-3, '> 0', '>=');
record_check('Q2.6', 'Q2', 'all-zero logits: |L - log V|', abs(cross_entropy(zeros(5, V), (1:5)') - log(V)), 1e-15);
record_check('Q2.7', 'Q2', 'target probability 0.7: |NLL + log 0.7|', abs(cross_entropy(log([0.7 0.2 0.1]), 1) + log(0.7)), 1e-15);
Hh = randn(12, 6);  Wh = 0.3 * randn(6, V);  yh = randi(V, 12, 1);
Gh = cross_entropy_backward(Hh * Wh, yh);
if has_dl
    [~, gHh, gWh] = dlfeval(@head_objective, dlarray(Hh), dlarray(Wh), yh);
    record_check('Q2.8', 'Q2', 'head gradients Hbar'' G and G W_voc'' vs dlgradient', ...
                 max([max(abs(extractdata(gWh) - Hh' * Gh), [], 'all'), max(abs(extractdata(gHh) - Gh * Wh'), [], 'all')]), 1e-15);
else
    E = zeros(size(Wh));  E(3, 5) = 1;
    fd = (cross_entropy(Hh * (Wh + 1e-5 * E), yh) - cross_entropy(Hh * (Wh - 1e-5 * E), yh)) / 2e-5;
    G_W = Hh' * Gh;
    record_check('Q2.8', 'Q2', 'head gradient Hbar'' G vs centred difference', rel_err(fd, G_W(3, 5)), 1e-9);
end
R.Q2 = struct('L', Lx, 'ref', ref, 'grad', Gx, 'L_col', L_col, 'L_row', L_row, 'L_mix', L_mix);

%% Case B1: fixed logit matrix and exact cross-entropy
print_section('Case B1  Fixed logit matrix and exact cross-entropy');
Zb1 = [0 0 0; log(2) 0 0; 0 log(3) 0];  yb1 = [1; 1; 2];
[Lb1, Pb1] = cross_entropy(Zb1, yb1);
Lref = (log(3) + log(2) + log(5/3)) / 3;
Eb1 = [1 0 0; 1 0 0; 0 1 0];
show('P = softmax(Z)', Pb1, '%10.7f');
record_check('B1.1', 'Q2', '|L - L_ref|', abs(Lb1 - Lref), 1e-6);
record_check('B1.2', 'Q2', 'max |dL/dZ - (P - E)/3|', max(abs(cross_entropy_backward(Zb1, yb1) - (Pb1 - Eb1) / 3), [], 'all'), 1e-6);
record_check('B1.3', 'Q2', 'correct-class probabilities vs (1/3, 1/2, 3/5)', max(abs(Pb1(sub2ind([3 3], 1:3, yb1')) - [1/3 1/2 3/5])), 1e-15);
record_check('B1.4', 'Q2', 'shifting each row by a constant: |delta L|', abs(cross_entropy(Zb1 + [5; -3; 100], yb1) - Lb1), 1e-13);
R.B1 = struct('P', Pb1, 'L', Lb1, 'ref', Lref);

%% Q3: automatic differentiation and Adam
print_section('Q3  Automatic differentiation and Adam');
st = adam_init(struct('x', 0));
[Fx, st] = adam_step(struct('x', 0), struct('x', 0.2), st, 1e-3, 0.9, 0.999, 1e-8);
upd = -1e-3 * 0.2 / (sqrt(0.999 * 0 + 0.001 * 0.04 / 0.001) + 1e-8);
fprintf('g1 = 0.2: m1 = %g, v1 = %g, update = %.12f\n', st.m.x, st.v.x, Fx.x);
record_check('Q3.1', 'Q3', 'scalar Adam exercise: |optimizer update - hand computation|', abs(Fx.x - upd), 1e-18);
rng(SEED, 'twister');
[Xr, Yr] = get_batch(train, B, T);
[Zr, cr] = gpt_forward(Xr, P0, eps_ln);
Lr = cross_entropy(Zr, Yr(:));
GZr = cross_entropy_backward(Zr, Yr(:));
gman = pack_params(gpt_backward(GZr, cr, P0));
fprintf('real batch at the Assignment 1 initialization: loss %.6f\n', Lr);
if has_dl
    [~, gdl] = loss_and_grad(pack_params(P0), Xr, Yr(:), L, eps_ln, 'dlgradient');
    record_check('Q3.2', 'Q3', 'every parameter gradient: hand-derived chain rule vs dlgradient', max_struct_diff(gman, gdl), 1e-14);
else
    fprintf('Deep Learning Toolbox not available: Q3.2 (dlgradient referee) skipped\n');
end
nzero = sum(cellfun(@(g) ~any(g(:) ~= 0), struct2cell(gman)));
record_check('Q3.3', 'Q3', 'tensors without any gradient', nzero, 0, '0', '==');
record_check('Q3.4', 'Q3', 'dL/dW_voc = Hbar'' dL/dZ: shape d x V and value', max(abs(cr.Hbar' * GZr - gman.W_voc), [], 'all') + ...
             double(~isequal(size(gman.W_voc), [d V])), 1e-16);
okE = all(isfinite(gman.W_E), 'all') && any(gman.W_E(:) ~= 0) && all(isfinite(gman.W_P), 'all') && any(gman.W_P(:) ~= 0);
record_check('Q3.5', 'Q3', 'W_E and W_P gradients: zero or non-finite (0 = both fine)', double(~okE), 0, '0', '==');
if has_dl
    F0d = structfun(@dlarray, pack_params(P0), 'UniformOutput', false);
    [~, gdet] = dlfeval(@detached_objective, F0d, Xr, Yr(:), L, eps_ln);
    record_check('Q3.6', 'Q3', 'gradient reaching W_E through extractdata (detached copy)', max(abs(extractdata(gdet.W_E)), [], 'all'), 0, '0', '==');
end
R.Q3 = struct('loss', Lr, 'norms', group_norms(gman));

%% Case B3 (part 1): one Adam step on a random 2 x 3 matrix
print_section('Case B3 (part 1)  One Adam step on a random 2 x 3 matrix (seed 451)');
rng(SEED, 'twister');
th0 = randn(2, 3);  gB3 = [0.5 -0.2 0.01; -1.5 0.3 2.0];
m1 = 0.1 * gB3;  v1 = 0.001 * gB3.^2;
th_hand = th0 - 1e-3 * (m1 / 0.1) ./ (sqrt(v1 / 0.001) + 1e-8);
stB3 = adam_init(struct('W', th0));
[FB3, stB3] = adam_step(struct('W', th0), struct('W', gB3), stB3, 1e-3, 0.9, 0.999, 1e-8);
show('theta_1', FB3.W, '%12.8f');
record_check('B3.1', 'Q4', 'Adam step: max |optimizer - matrix equations| (m1, v1, theta1)', ...
             max([max(abs(stB3.m.W - m1), [], 'all'), max(abs(stB3.v.W - v1), [], 'all'), max(abs(FB3.W - th_hand), [], 'all')]), 1e-6);

%% Benchmark B
print_section('Benchmark B  Deterministic logits, loss, gradients and Adam (B=1, T=2, d=2, V=3)');
Hbb = reshape(eye(2), 1, 2, 2);
Rbb = Hbb + zeros(1, 2, 2);
Gbb = ffn_forward(layernorm_forward(Rbb, [1 1], [0 0], eps_ln), zeros(2, 8), zeros(1, 8), zeros(8, 2), zeros(1, 2));
record_check('BB.1', 'Bench', 'max |R - H| + |G| + |H'' - H|', max(abs(Rbb - Hbb), [], 'all') + max(abs(Gbb), [], 'all') + ...
             max(abs(Rbb + Gbb - Hbb), [], 'all'), 0, '0', '==');
Hbar_b = eye(2);  W_b = zeros(2, 3);  y_b = [2; 3];
Z_b = Hbar_b * W_b;
[L_b, P_b] = cross_entropy(Z_b, y_b);
GZ_b = cross_entropy_backward(Z_b, y_b);
GW_b = Hbar_b' * GZ_b;
GW_ref = [1/6 -1/3 1/6; 1/6 1/6 -1/3];
show('P', P_b, '%10.7f');  show('dL/dW_voc', GW_b, '%10.6f');
record_check('BB.2', 'Bench', 'logits, probabilities, loss: max abs error', max([max(abs(Z_b), [], 'all'), max(abs(P_b - 1/3), [], 'all'), abs(L_b - log(3))]), 1e-6);
record_check('BB.3', 'Bench', 'dL/dZ and dL/dW_voc vs the displayed matrices', max([max(abs(GZ_b - GW_ref), [], 'all'), max(abs(GW_b - GW_ref), [], 'all')]), 1e-6);
stb = adam_init(struct('W', W_b));
Fb = adam_step(struct('W', W_b), struct('W', GW_b), stb, 1e-3, 0.9, 0.999, 1e-8);
show('W_voc after one Adam step', Fb.W, '%12.9f');
record_check('BB.4', 'Bench', 'first Adam update vs -eta g/(|g| + eps)', max(abs(Fb.W - (-1e-3 * GW_b ./ (abs(GW_b) + 1e-8))), [], 'all'), 1e-7);
record_check('BB.5', 'Bench', 'first Adam update vs the rounded display (+-0.001)', max(abs(Fb.W - 1e-3 * [-1 1 -1; -1 -1 1]), [], 'all'), 1e-7);
R.BenchB = struct('P', P_b, 'L', L_b, 'GZ', GZ_b, 'GW', GW_b, 'W_new', Fb.W);

%% Q4: training schedule and gradient checks
print_section('Q4  Training schedule and gradient checks');
s_epoch = floor(numel(train) / (B * T));
fprintf('S_epoch = floor(%d / %d) = %d; S_total = 50 S_epoch = %d\n', numel(train), B * T, s_epoch, 50 * s_epoch);
record_check('Q4.1', 'Q4', 'S_epoch - floor(N_train / (B T))', s_epoch - floor(numel(train) / (B * T)), 0, '0', '==');
rng(SEED, 'twister');
[X4, Y4] = get_batch(train, 4, T);                  % fixed batch: no resampling during the test
F4 = pack_params(P0);
[~, G4] = loss_and_grad(F4, X4, Y4(:), L, eps_ln, 'manual');
names = {'W_E', 'W_P', 'b1_W_QKV', 'b1_W_O', 'b1_W_1', 'b1_b_1', 'b1_W_2', 'b2_b_2', 'b1_ln1_gamma', 'b2_ln2_beta', 'lnf_gamma', 'W_voc'};
deltas = [1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8];
rho = zeros(numel(names), numel(deltas));
for k = 1:numel(names)
    g = G4.(names{k});
    [~, ik] = max(abs(g(:)));
    for j = 1:numel(deltas)
        Fp = F4;  Fm = F4;
        Fp.(names{k})(ik) = Fp.(names{k})(ik) + deltas(j);
        Fm.(names{k})(ik) = Fm.(names{k})(ik) - deltas(j);
        fd = (cross_entropy(gpt_forward(X4, unpack_params(Fp, L), eps_ln), Y4(:)) - ...
              cross_entropy(gpt_forward(X4, unpack_params(Fm, L), eps_ln), Y4(:))) / (2 * deltas(j));
        rho(k, j) = abs(fd - g(ik)) / max([1, abs(fd), abs(g(ik))]);
    end
    fprintf('   %-14s %s\n', names{k}, sprintf('%8.0e', rho(k, :)));
end
[best_rho, ib] = min(max(rho, [], 1));
record_check('Q4.2', 'Q4', 'full-model rho_k at delta = 1e-5 (12 tensors, fixed batch)', max(rho(:, 4)), 1e-8);
record_check('Q4.3', 'Q4', sprintf('full-model rho_k at the best delta (%g)', deltas(ib)), best_rho, 1e-9);
record_check('Q4.4', 'Q4', 'non-finite parameters or gradients at the fixed batch', ...
             double(~all(cellfun(@(g) all(isfinite(g), 'all'), [struct2cell(F4); struct2cell(G4)]))), 0, '0', '==');
R.Q4 = struct('s_epoch', s_epoch, 'rho', rho, 'deltas', deltas, 'best_delta', deltas(ib));

%% Case B3 (part 2): the complete model on ABAB...
print_section('Case B3 (part 2)  The complete model on a repeated synthetic sequence ABAB...');
syn = repmat('AB', 1, 2000);
vsyn = build_vocab(syn);
xsyn = encode_text(syn, vsyn);
[trs, vas] = split_contiguous(xsyn);
rng(SEED, 'twister');
Fsyn = pack_params(init_params(2, 32, 32, 2, 0.02));
cfg_syn = CFG;  cfg_syn.B = 16;  cfg_syn.T = 32;  cfg_syn.epochs = 100;  cfg_syn.s_max = 300;
cfg_syn.eval_every = 10;  cfg_syn.preview_every = 0;  cfg_syn.grad = 'manual';
streams_syn = struct('train', RandStream('mt19937ar', 'Seed', SEED + 1), 'val', RandStream('mt19937ar', 'Seed', SEED + 2), ...
                     'preview', RandStream('mt19937ar', 'Seed', SEED + 3));
[Fsyn, rsyn] = train_gpt(Fsyn, trs, vas, cfg_syn, streams_syn, vsyn);
greedy_syn = generate_text(unpack_params(Fsyn, 2), 'A', vsyn, 40, 32, eps_ln, [], 1, true);
fprintf('training loss %.4f -> %.6f; greedy continuation %s\n', rsyn.train_curve(1), rsyn.train_curve(end), greedy_syn);
record_check('B3.2', 'Q4', 'ABAB: final training loss / initial', rsyn.train_curve(end) / rsyn.train_curve(1), 0.05);
record_check('B3.3', 'Q4', 'ABAB greedy continuation: characters that break the alternation', sum(greedy_syn(1:end-1) == greedy_syn(2:end)), 0, '0', '==');
R.B3 = struct('first', rsyn.train_curve(1), 'last', rsyn.train_curve(end), 'greedy', greedy_syn);

%% Q6: integration (shapes, residual input, masked weights, logits)
print_section('Q6  Integration: shapes, residual input, masked weights, logits');
rows = param_table(P0);
for k = 1:size(rows, 1)
    fprintf('%-24s %-10s %8d\n', rows{k, 1}, mat2str(rows{k, 2}), rows{k, 3});
end
record_check('Q6.1', 'Q6', 'P - (128V + 103680)', count_params(P0) - (128 * V + 103680), 0, '0', '==');
b1c = cr.blocks{1};
record_check('Q6.2', 'Q6', 'embedding output and block input are B x T x d (mismatch count)', ...
             double(~isequal(size(b1c.H), [B T d])) + double(~isequal(size(b1c.U), [B T d])), 0, '0', '==');
record_check('Q6.3', 'Q6', 'first residual: max |R - (H0 + F)|', max(abs(b1c.R - (b1c.H + b1c.F)), [], 'all'), 0, '0', '==');
fut = triu(true(T), 1);
Amax = 0;  Arow = 0;
for l = 1:L
    A = cr.blocks{l}.att.A;                          % T x T x B pages
    Amax = max(Amax, max(A(repmat(fut, 1, 1, B))));
    Arow = max(Arow, max(abs(sum(A, 2) - 1), [], 'all'));
end
record_check('Q6.4', 'Q6', 'both blocks: largest masked future weight', Amax, 1e-7);
record_check('Q6.5', 'Q6', 'both blocks: max_i |sum_j A_ij - 1|', Arow, 1e-12);
record_check('Q6.6', 'Q6', 'logits shape (BT x V) mismatch + non-finite loss', double(~isequal(size(Zr), [B*T V])) + double(~isfinite(Lr)), 0, '0', '==');
record_check('Q6.7', 'Q6', 'initial loss vs log V: |L0 - log V|', abs(Lr - log(V)), 0.05);
R.Q6 = struct('L0', Lr, 'logV', log(V));

%% Minimum acceptance tests
print_section('Minimum acceptance tests');
acc = [acc_case('one-token context T=1 (B=3)', [3; 1; 2], 5, eps_ln), ...
       acc_case('batch of size one B=1 (T=6)', [2 4 1 3 3 5], 5, eps_ln), ...
       acc_case('repeated character', ones(1, 7), 3, eps_ln), ...
       acc_case('tiny vocabulary V=2', [1 2 2 1; 2 1 1 2], 2, eps_ln), ...
       acc_case('short batch B=2, T=3', [1 2 3; 3 3 2], 3, eps_ln)];
record_check('ACC.1', 'Q6', 'non-finite losses or gradients in the five cases', sum(~[acc.finite]), 0, '0', '==');
record_check('ACC.2', 'Q6', 'change the last token: max change of earlier positions'' logits', max([acc.causal_delta]), 0, '0', '==');
last = [acc.last_delta];
record_check('ACC.3', 'Q6', 'the change reaches the last position (min over cases with T > 1)', min(last(last >= 0)), 1e-12, '> 0', '>=');
record_check('ACC.4', 'Q6', 'vectorized vs per-sample attention (max over cases)', max([acc.vec_vs_loop]), 1e-14);
R.acceptance = acc;

%% Case B5: Assignment 1 standalone attention vs the attention inside Assignment 2
print_section('Case B5  Assignment 1 attention vs the attention inside the Assignment 2 model');
rng(SEED, 'twister');
[Xv5, ~, rv5] = get_batch(val, B, T);               % held-out batch, Assignment 1 protocol
H05 = embed_forward(Xv5, P0.W_E, P0.W_P);
U5 = layernorm_forward(H05, P0.blocks(1).ln1_gamma, P0.blocks(1).ln1_beta, eps_ln);
F_a1 = attention_forward(U5, P0.blocks(1).W_QKV, P0.blocks(1).W_O);
F_loop = attention_per_sample(U5, P0.blocks(1).W_QKV, P0.blocks(1).W_O);
[~, c5] = gpt_forward(Xv5, P0, eps_ln);
e5 = max([max(abs(F_a1 - c5.blocks{1}.F), [], 'all'), max(abs(F_a1 - F_loop), [], 'all')]);
fprintf('validation starts %s ...; max |F(A1) - F(inside A2)|, |F - per-sample| = %.2e\n', mat2str(rv5(1:5)'), e5);
record_check('B5.1', 'Q6', 'standalone (A1) vs embedded (A2) attention, saved weights', e5, 1e-5);

%% Case B4: Tiny Shakespeare end-to-end training
print_section('Case B4  Tiny Shakespeare end-to-end training (B=32, T=64, d=64, L=2, lr 1e-3, 50 epochs)');
streams = struct('train', RandStream('mt19937ar', 'Seed', SEED), 'val', RandStream('mt19937ar', 'Seed', SEED + 1), ...
                 'preview', RandStream('mt19937ar', 'Seed', SEED + 2));
[Ftr, rec] = train_gpt(pack_params(P0), train, val, CFG, streams, vocab);
Ptr = unpack_params(Ftr, L);
save(fullfile(out_dir, 'model_final.mat'), 'Ptr', 'vocab', 'CFG');
writetable(array2table(rec.evals, 'VariableNames', {'step', 'train_loss', 'val_loss'}), fullfile(out_dir, 'training_log.csv'));
sv = unique(round(linspace(1, numel(val) - T, CFG.fixed_eval_windows)));
st_ = unique(round(linspace(1, numel(train) - T, CFG.fixed_eval_windows)));
L_va = fixed_loss(Ptr, val, sv, T, eps_ln);
L_tr = fixed_loss(Ptr, train, st_, T, eps_ln);
L_va0 = fixed_loss(P0, val, sv, T, eps_ln);
fprintf('fixed validation windows: initial %.4f (log V %.4f) -> final %.4f; training windows %.4f; gap %+.4f\n', ...
        L_va0, log(V), L_va, L_tr, L_va - L_tr);
record_check('B4.1', 'Q4', 'training steps taken - S_total', rec.adam_steps - rec.s_total, 0, '0', '==');
record_check('B4.2', 'Q4', 'validation only from the held-out part, no update during validation (1 = ok)', double(rec.val_ok), 1, '1', '==');
record_check('B4.3', 'Q4', 'final fixed-window validation loss below log V (nats)', log(V) - L_va, 0.5, '> 0.5', '>=');
record_check('B4.4', 'Q4', 'non-finite trained parameters', double(~all(cellfun(@(g) all(isfinite(g), 'all'), struct2cell(Ftr)))), 0, '0', '==');
R.B4 = struct('s_epoch', rec.s_epoch, 's_total', rec.s_total, 'seconds', rec.seconds, 'L_val_fixed', L_va, ...
              'L_train_fixed', L_tr, 'L_val_fixed_init', L_va0, 'gap', L_va - L_tr, 'evals', rec.evals, ...
              'norms', rec.norms, 'previews', rec.previews, 'grad', CFG.grad);

%% Q5: autoregressive generation
print_section('Q5  Autoregressive generation');
zq = [2 1 0];
pq = exp(zq - max(zq));  pq = pq / sum(pq);
record_check('Q5.1', 'Q5', 'z=(2,1,0): |p - (e^2, e, 1)/(e^2+e+1)| and |sum p - 1|', max([max(abs(pq - exp(zq) / sum(exp(zq)))), abs(sum(pq) - 1)]), 1e-15);
rs5 = RandStream('mt19937ar', 'Seed', SEED);
cdf = cumsum(pq);
draws = arrayfun(@(k) find(cdf >= rand(rs5), 1), 1:100000);
freq = histcounts(draws, 0.5:1:3.5) / 100000;
record_check('Q5.2', 'Q5', 'sampler: max |empirical frequency - p| over 100000 draws', max(abs(freq - pq)), 0.01);
prompt_ids = x(1001:1000 + T + 5);
Zc = gpt_forward(prompt_ids(6:end), Ptr, eps_ln);
Zk = gpt_forward(prompt_ids(end - T + 1:end), Ptr, eps_ln);
fprintf('prompt length T + 5 = %d: indices 6..%d are kept and re-indexed as positions 1..%d\n', numel(prompt_ids), T + 5, T);
record_check('Q5.3', 'Q5', 'cropped context: explicit crop vs the last T tokens', max(abs(Zc(end, :) - Zk(end, :))), 0, '0', '==');
kept = prompt_ids(6:end);  kept2 = kept;  kept2(end) = mod(kept2(end), V) + 1;
Z2 = gpt_forward(kept2, Ptr, eps_ln);
record_check('Q5.4', 'Q5', 'changing the newest token: change of all earlier rows'' logits', max(abs(Z2(1:end-1, :) - Zc(1:end-1, :)), [], 'all'), 0, '0', '==');
txt = generate_text(Ptr, CFG.prompt, vocab, CFG.sample_len, T, eps_ln, RandStream('mt19937ar', 'Seed', SEED + 3), 1, false);
greedy = generate_text(Ptr, CFG.prompt, vocab, CFG.sample_len, T, eps_ln, [], 1, true);
fprintf('sampled continuation (200 characters, tau = 1):\n%s\n\ngreedy continuation:\n%s\n', txt, greedy);
record_check('Q5.5', 'Q5', 'generated characters outside the corpus vocabulary', sum(~ismember(txt, vocab)), 0, '0', '==');
record_check('Q5.6', 'Q5', 'sample length - (prompt + 200)', numel(txt) - (numel(CFG.prompt) + CFG.sample_len), 0, '0', '==');
fid = fopen(fullfile(out_dir, 'samples.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, 'prompt: ''%s''\n\n[categorical sampling, tau = 1, 200 characters]\n%s\n\n[greedy, for contrast]\n%s\n', CFG.prompt, txt, greedy);
fclose(fid);
R.Q5 = struct('p', pq, 'freq', freq, 'sample', txt, 'greedy', greedy);

%% Figure: training and validation loss
plot_loss_curves(rec.evals, log(V), L_va, fullfile(fig_dir, 'loss_curves_matlab.png'));

%% Summary and outputs
print_section('Summary');
rows_chk = record_check('all');
n_pass = sum([rows_chk.passed]);
fprintf('checks passed: %d / %d\nwall time %.1f s\n', n_pass, numel(rows_chk), toc(t_start));
R.config = CFG;
R.checks = rows_chk;
R.summary = struct('passed', n_pass, 'total', numel(rows_chk), 'seconds', toc(t_start));
fid = fopen(fullfile(out_dir, 'results.json'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true));
fclose(fid);
writetable(struct2table(rows_chk), fullfile(out_dir, 'checks.csv'));
diary off;

%% ---------------------------------------------------------------- local functions
function print_section(title_text)
fprintf('\n%s\n%s\n%s\n', repmat('=', 1, 78), title_text, repmat('=', 1, 78));
end

function show(name, A, fmt)
fprintf('%s  size=%s\n', name, mat2str(size(A)));
for i = 1:size(A, 1)
    fprintf('   [%s ]\n', sprintf([' ' fmt], A(i, :)));
end
end

function m = max_struct_diff(A, Bs)
% largest absolute difference over every field of two flat structs with the same fields
fn = fieldnames(A);
m = 0;
for k = 1:numel(fn)
    m = max(m, max(abs(A.(fn{k}) - Bs.(fn{k})), [], 'all'));
end
end

function P = small_params(V, T, d, L)
% a small random model with non-trivial gains and biases (uses the global stream)
P = init_params(V, T, d, L, 0.3);
for l = 1:L
    fn = fieldnames(P.blocks(l));
    for k = 1:numel(fn)
        if contains(fn{k}, 'gamma') || contains(fn{k}, 'beta') || startsWith(fn{k}, 'b_')
            P.blocks(l).(fn{k}) = P.blocks(l).(fn{k}) + 0.1 * randn(size(P.blocks(l).(fn{k})));
        end
    end
end
end

function [Rr, F] = residual_branch(H, blk, eps_ln)
% first residual of a block: R = H + attention(LN1(H))
U = layernorm_forward(H, blk.ln1_gamma, blk.ln1_beta, eps_ln);
F = attention_forward(U, blk.W_QKV, blk.W_O);
Rr = H + F;
end

function [J, g] = gelu_sum(z)
J = sum(gelu_tanh(z), 'all');
g = dlgradient(J, z);
end

function [J, gW1, gb1, gW2, gb2, gD] = ffn_objective(D, W1, b1, W2, b2, Rg)
J = sum(ffn_forward(D, W1, b1, W2, b2) .* Rg, 'all');
[gW1, gb1, gW2, gb2, gD] = dlgradient(J, W1, b1, W2, b2, D);
end

function [J, gH, gW] = head_objective(Hbar, W, y)
J = cross_entropy(Hbar * W, y);
[gH, gW] = dlgradient(J, Hbar, W);
end

function [J, G] = detached_objective(F, X, y, L, eps_ln)
% W_E enters through extractdata: a numerical copy without history
P = unpack_params(F, L);
P.W_E = extractdata(P.W_E);
J = cross_entropy(gpt_forward(X, P, eps_ln), y);
G = dlgradient(J, F);
end

function Lf = fixed_loss(P, ids, starts, T, eps_ln)
% mean loss over fixed, documented windows, in chunks of 64 (equal weight per character)
tot = 0;  cnt = 0;
for i = 1:64:numel(starts)
    r = starts(i:min(i + 63, numel(starts)))';
    idx = r + (0:T-1);
    X = reshape(ids(idx), numel(r), T);
    Y = reshape(ids(idx + 1), numel(r), T);
    tot = tot + cross_entropy(gpt_forward(X, P, eps_ln), Y(:)) * numel(X);
    cnt = cnt + numel(X);
end
Lf = tot / cnt;
end

function row = acc_case(name, X, V, eps_ln)
% one acceptance case: finite loss and gradients, causality of the logits, vectorized vs per-sample attention
rng(451, 'twister');
[Bc, Tc] = size(X);
P = small_params(V, Tc, 8, 2);
Y = randi(V, Bc, Tc);
F = pack_params(P);
[Lc, G] = loss_and_grad(F, X, Y(:), 2, eps_ln, 'manual');
finite = isfinite(Lc) && all(cellfun(@(g) all(isfinite(g), 'all'), struct2cell(G)));
[Z1, c] = gpt_forward(X, P, eps_ln);
causal = 0;  last = -1;
if Tc > 1
    X2 = X;  X2(:, end) = mod(X2(:, end), V) + 1;
    Z2 = gpt_forward(X2, P, eps_ln);
    Z1r = reshape(Z1, Bc, Tc, V);  Z2r = reshape(Z2, Bc, Tc, V);     % column-major rows: r = b + (t-1) B
    causal = max(abs(Z2r(:, 1:end-1, :) - Z1r(:, 1:end-1, :)), [], 'all');
    last = max(abs(Z2r(:, end, :) - Z1r(:, end, :)), [], 'all');
end
F_loop = attention_per_sample(c.blocks{1}.U, P.blocks(1).W_QKV, P.blocks(1).W_O);
row = struct('name', name, 'finite', finite, 'loss', Lc, 'causal_delta', causal, 'last_delta', last, ...
             'vec_vs_loop', max(abs(c.blocks{1}.F - F_loop), [], 'all'));
fprintf('%-34s loss %.4f finite %d causal %.1e last %.1e vec-loop %.1e\n', name, Lc, finite, causal, last, row.vec_vs_loop);
end
