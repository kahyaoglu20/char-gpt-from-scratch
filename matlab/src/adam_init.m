function state = adam_init(F)
%ADAM_INIT Zero first and second moments for every field of the flat parameter struct F.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
state = struct('t', 0, 'm', structfun(@(x) zeros(size(x)), F, 'UniformOutput', false), ...
               'v', structfun(@(x) zeros(size(x)), F, 'UniformOutput', false));
end
