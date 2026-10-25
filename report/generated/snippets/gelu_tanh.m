function y = gelu_tanh(z)
%GELU_TANH Elementwise tanh approximation of GELU:
%   g(z) = z/2 * (1 + tanh(sqrt(2/pi) * (z + 0.044715 z^3))).
%   Works on numeric arrays and on traced dlarray inputs.
y = 0.5 * z .* (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z.^3)));
end
