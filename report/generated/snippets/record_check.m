function out = record_check(id, question, description, observed, threshold, expected, rule)
%RECORD_CHECK Store and print one verification row (persistent list).
%   record_check(id, question, description, observed, threshold, expected, rule)
%   rows = record_check('all');   record_check('reset');
persistent ROWS
if isempty(ROWS)
    ROWS = struct('id', {}, 'question', {}, 'description', {}, 'expected', {}, ...
                  'observed', {}, 'threshold', {}, 'rule', {}, 'passed', {});
end
if nargin == 1
    if strcmp(id, 'reset')
        ROWS = ROWS([]);
    end
    out = ROWS;
    return
end
if nargin < 6, expected = '0'; end
if nargin < 7, rule = '<='; end
observed = double(observed);
switch rule
    case '<=', passed = observed <= threshold;
    case '>=', passed = observed >= threshold;
    case '==', passed = observed == threshold;
    otherwise, error('record_check:rule', 'unknown rule %s', rule);
end
ROWS(end+1) = struct('id', id, 'question', question, 'description', description, ...
    'expected', expected, 'observed', observed, 'threshold', threshold, 'rule', rule, ...
    'passed', logical(passed));
tags = {'FAIL', 'PASS'};
fprintf('[%s] %s: %s = %.3e (expected %s, pass if %s %g)\n', tags{passed + 1}, id, ...
        description, observed, expected, rule, threshold);
out = ROWS;
end
