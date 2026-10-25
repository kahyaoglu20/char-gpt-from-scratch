function plot_loss_curves(evals, logV, L_val_fixed, file)
%PLOT_LOSS_CURVES Training loss (measured before each update) and validation loss (after it) at
%   the evaluation steps, with the uniform level log V and the final fixed-window validation loss.
%   evals is the (step, training loss, validation loss) record of train_gpt; saved with exportgraphics.
fig = figure('Visible', 'off', 'Position', [100 100 900 420]);
ax = axes(fig);
h1 = plot(ax, evals(:, 1), evals(:, 2), '-', 'Color', [0.17 0.42 0.69]);
hold(ax, 'on');
h2 = plot(ax, evals(:, 1), evals(:, 3), '-', 'Color', [0.87 0.42 0.13]);
h3 = yline(ax, logV, '--k');
h4 = yline(ax, L_val_fixed, ':', 'Color', [0.70 0.15 0.35], 'LineWidth', 1.5);
ax.XAxis.Exponent = 0;                              % plain step numbers, no x10^4 multiplier
xlim(ax, [0 evals(end, 1)]);
ylim(ax, [1 4.5]);
xlabel(ax, 'optimizer step');
ylabel(ax, 'cross-entropy (nats per character)');
grid(ax, 'on');
legend(ax, [h1 h2 h3 h4], {'training batch (before update)', 'validation batch, 8 windows (after update)', ...
       sprintf('uniform prediction log V = %.3f', logV), sprintf('final validation, fixed windows = %.3f', L_val_fixed)}, ...
       'Location', 'east');
title(ax, 'Tiny Shakespeare: training and held-out loss (MATLAB)');
exportgraphics(fig, file, 'Resolution', 200);
close(fig);
end
