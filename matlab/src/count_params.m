function n = count_params(P)
%COUNT_PARAMS Programmatic count: number of scalar entries of all tensors.
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
rows = param_table(P);
n = sum(cell2mat(rows(:, 3)));
end
