function [text, raw] = load_corpus(path)
%LOAD_CORPUS Read the corpus exactly as stored on disk (no newline translation).
%   [text, raw] = load_corpus(path) returns the UTF-8 decoded character row
%   vector and the raw bytes (uint8 row vector) used for the SHA-256 checksum.
fid = fopen(path, 'r');
if fid < 0
    error('load_corpus:missing', 'cannot open %s', path);
end
raw = fread(fid, inf, '*uint8')';        % exact bytes, 1 x nbytes
fclose(fid);
text = native2unicode(raw, 'UTF-8');     % 1 x N characters, whitespace preserved
end
