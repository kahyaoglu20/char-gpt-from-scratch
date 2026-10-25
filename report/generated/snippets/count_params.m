function n = count_params(P)
%COUNT_PARAMS Programmatic count: number of scalar entries of all tensors.
rows = param_table(P);
n = sum(cell2mat(rows(:, 3)));
end
