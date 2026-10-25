function x = encode_text(text, vocab)
%ENCODE_TEXT x_i = f(s_i): position of every character in the sorted vocabulary.
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
[found, x] = ismember(text, vocab);      % x(i) in 1..V
if ~all(found)
    error('encode_text:oov', 'character outside the vocabulary');
end
x = double(x);                           % 1 x N index vector
end
