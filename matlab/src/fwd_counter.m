function out = fwd_counter(action, name)
%FWD_COUNTER Count tested forward evaluations of each component.
%   fwd_counter('inc', 'attention'); c = fwd_counter('get'); fwd_counter('reset').
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
persistent C
if isempty(C)
    C = struct('embed', 0, 'layernorm', 0, 'attention', 0);
end
switch action
    case 'inc'
        C.(name) = C.(name) + 1;
    case 'reset'
        C = struct('embed', 0, 'layernorm', 0, 'attention', 0);
end
out = C;
end
