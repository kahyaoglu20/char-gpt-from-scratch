function gp = gelu_grad(z)
%GELU_GRAD The assignment's derivative of the tanh-approximated GELU:
%   g'(z) = (1 + tanh u)/2 + z/2 (1 - tanh^2 u) a (1 + 3 c z^2),  u = a (z + c z^3),
%   a = sqrt(2/pi), c = 0.044715.
%   MAT 451E Assignment 2 - Ömer Kahyaoğlu (090240741)
a = sqrt(2/pi);
c = 0.044715;
t = tanh(a * (z + c * z.^3));
gp = 0.5 * (1 + t) + 0.5 * z .* (1 - t.^2) * a .* (1 + 3 * c * z.^2);
end
