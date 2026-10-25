function rows = param_table(P)
%PARAM_TABLE Flatten the parameter struct into {name, size, numel, init} rows.
rows = {};
names = fieldnames(P);
for k = 1:numel(names)
    v = P.(names{k});
    if isstruct(v)
        for l = 1:numel(v)
            sub = fieldnames(v(l));
            for j = 1:numel(sub)
                rows(end+1, :) = describe(sprintf('blocks(%d).%s', l, sub{j}), v(l).(sub{j})); %#ok<AGROW>
            end
        end
    else
        rows(end+1, :) = describe(names{k}, v); %#ok<AGROW>
    end
end
end

function row = describe(name, value)
if contains(name, 'gamma')
    rule = 'ones';
elseif contains(name, 'beta') || contains(name, '.b_')
    rule = 'zeros';
else
    rule = 'N(0, 0.02^2)';
end
row = {name, size(value), numel(value), rule, value};
end
