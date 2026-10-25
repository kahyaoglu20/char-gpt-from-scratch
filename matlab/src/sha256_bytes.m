function hex = sha256_bytes(raw)
%SHA256_BYTES SHA-256 checksum (lower-case hex) of a uint8 vector, via Java.
%   MAT 451E Assignment 1 - Ömer Kahyaoğlu (090240741)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(raw(:));
digest = typecast(md.digest(), 'uint8');
hex = lower(reshape(dec2hex(digest, 2)', 1, []));
end
