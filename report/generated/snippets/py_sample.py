def next_distribution(context, P, T, L=2, temperature=1.0):
    """p over the vocabulary for the next character, given the full history `context` (1-D indices)."""
    window = np.asarray(context)[-T:]                       # crop: the newest T tokens, positions re-indexed 0..len-1
    Z, _ = gpt_forward(window[None, :], P, L)               # (len(window) x V) logits for a batch of one
    z_last = Z.data[-1] / temperature                       # only the final row predicts the next character
    return softmax_np(z_last[None, :])[0]

def sample_index(p, rng):
    """Inverse-CDF categorical draw: the first index whose cumulative probability exceeds u ~ U(0,1)."""
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(p) - 1))

def generate(P, prompt, stoi, itos, n_new, T, rng, L=2, temperature=1.0, greedy=False):
    """Encode `prompt`, append `n_new` sampled (or greedy) characters, decode the whole sequence."""
    Pc = as_constants(P)
    ids = list(encode(prompt, stoi))
    for _ in range(n_new):
        p = next_distribution(ids, Pc, T, L, temperature)
        ids.append(int(np.argmax(p)) if greedy else sample_index(p, rng))
    return decode(np.array(ids), itos)
