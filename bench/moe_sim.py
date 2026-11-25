"""Offline MoE expert-cache simulator for llama.cpp LLAMA_MOE_CACHE_TRACE files (format "MCT1",
src/llama-moe-cache.cpp:642). Usage:  python -I moe_sim.py trace.bin [--decode-only]
Reports, per cache size: hit rate and SSD/RAM bytes per token for LRU, LFU, static-hot and Belady (optimal).
Also: rho = share of a token's experts already used by the previous token (same layer)."""
import struct, sys
from collections import OrderedDict, Counter, defaultdict


def read_trace(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"MCT1", "not an MCT1 trace"
    o = 4
    def u(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o); o += struct.calcsize(fmt); return v
    n_layer, n_used, n_group = u("3I")
    expert_bytes, n_expert = {}, 0
    for _ in range(n_group):
        _dev, n_expert, n_lay = u("3I")
        lays = u(f"{n_lay}I")
        u("2d"); u("3Q")
        n_t, = u("I")
        eb = sum(u(f"{n_t}Q"))
        for l in lays:
            expert_bytes[l] = eb
    recs = []  # (layer, n_used, ids)
    while o < len(b):
        il, nu, n = u("3i")
        recs.append((il, nu, u(f"{n}i")))
    return n_layer, n_used, n_expert, expert_bytes, recs


def steps(recs, decode_only):
    """One step = one record per layer. Yields (layer, set_of_experts, n_tokens)."""
    for il, nu, ids in recs:
        nt = len(ids) // max(nu, 1)
        if decode_only and nt != 1:
            continue
        yield il, frozenset(i for i in ids if i >= 0), nt


def simulate(seq, slots, policy, future=None):
    """seq: list of (layer, experts). Per-layer cache of `slots` experts. Returns (hits, accesses)."""
    caches, hits, acc = defaultdict(OrderedDict), 0, 0
    freq = defaultdict(Counter)
    for t, (il, exps) in enumerate(seq):
        c = caches[il]
        for e in exps:
            acc += 1; freq[il][e] += 1
            if e in c:
                hits += 1
                if policy == "lru": c.move_to_end(e)
                continue
            if len(c) >= slots:
                if policy == "lru":
                    c.popitem(last=False)
                elif policy == "lfu":
                    del c[min([k for k in c if k not in exps] or list(c), key=lambda k: freq[il][k])]  # set larger than cache: evict among current set
                elif policy == "belady":  # evict the one used furthest in the future
                    nxt = future[t]
                    del c[max([k for k in c if k not in exps] or list(c), key=lambda k: nxt.get((il, k), 1 << 60))]
            c[e] = True
    return hits, acc


def next_use(seq):
    """future[t][(layer,e)] = index of next access after t (for Belady)."""
    out, last = [None] * len(seq), {}
    for t in range(len(seq) - 1, -1, -1):
        out[t] = dict(last)
        for e in seq[t][1]:
            last[(seq[t][0], e)] = t
    return out


def static_hot(seq, slots):
    """Pin the `slots` most frequent experts per layer (oracle frequency)."""
    freq = defaultdict(Counter)
    for il, exps in seq:
        freq[il].update(exps)
    pinned = {il: {e for e, _ in c.most_common(slots)} for il, c in freq.items()}
    acc = sum(len(x) for _, x in seq)
    hits = sum(len(x & pinned[il]) for il, x in seq)
    return hits, acc


def rho(seq):
    prev, same, tot = {}, 0, 0
    for il, exps in seq:
        if il in prev:
            same += len(exps & prev[il]); tot += len(exps)
        prev[il] = exps
    return same / tot if tot else 0.0


def main(path, decode_only=False):
    n_layer, n_used, n_expert, eb, recs = read_trace(path)
    st = list(steps(recs, decode_only))
    seq = [(il, ex) for il, ex, _ in st]
    n_tok = sum(nt for il, _, nt in st if il == st[0][0]) if st else 0
    mb = sum(eb.values()) / len(eb) / 2**20 if eb else 0
    print(f"layers={n_layer} experts={n_expert} top-k={n_used} expert={mb:.2f} MiB  records={len(seq)} tokens~{n_tok}")
    print(f"rho (expert reuse vs previous step, same layer) = {rho(seq):.3f}   random baseline = {n_used/n_expert:.3f}")
    fut = next_use(seq)
    acc = sum(len(x) for _, x in seq); uniq = len({(il, e) for il, x in seq for e in x})
    print(f"cold-start ceiling: {1 - uniq/acc:.3f} hit rate (unique experts touched = {uniq} = {uniq*mb/1024:.2f} GiB). 'static' is pre-warmed, others start cold.")
    print(f"{'slots/layer':>11} {'cache GiB':>9} | {'LRU':>6} {'LFU':>6} {'static':>6} {'Belady':>6} | miss GiB/token (LRU)")
    for frac in (0.125, 0.25, 0.375, 0.5, 0.75):
        s = max(n_used, int(n_expert * frac))
        r = {p: simulate(seq, s, p, fut) for p in ("lru", "lfu", "belady")}
        r["static"] = static_hot(seq, s)
        h = {p: v[0] / v[1] for p, v in r.items()}
        miss_bytes = (r["lru"][1] - r["lru"][0]) * mb / 1024 / max(n_tok, 1)
        print(f"{s:>11} {s*mb*len(eb)/1024:>9.2f} | {h['lru']:6.3f} {h['lfu']:6.3f} {h['static']:6.3f} {h['belady']:6.3f} | {miss_bytes:.3f}")


def _selfcheck():
    # 2 layers, 4 experts, top-2; layer 0 always uses {0,1} -> after warm-up every policy hits.
    seq = [(0, frozenset({0, 1})), (1, frozenset({2, 3}))] * 10
    for p in ("lru", "lfu", "belady"):
        h, a = simulate(seq, 2, p, next_use(seq))
        assert (h, a) == (36, 40), (p, h, a)
    assert static_hot(seq, 2) == (40, 40)
    assert abs(rho(seq) - 1.0) < 1e-9
    # Belady must never lose to LRU on a cyclic pattern that defeats LRU.
    cyc = [(0, frozenset({i % 3})) for i in range(30)]
    assert simulate(cyc, 2, "belady", next_use(cyc))[0] > simulate(cyc, 2, "lru")[0] == 0
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selfcheck()
    else:
        main(sys.argv[1], "--decode-only" in sys.argv)
