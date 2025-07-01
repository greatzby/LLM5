# evaluate_fixed.py  ---------------------------------------------------------
import torch, random
from collections import defaultdict

@torch.no_grad()
def evaluate_ar(model, test_file, stages,
                stoi, itos, device, graph,
                temperature=0.1, top_k=10,
                max_eval_per_type=50):
    """
    统一的 AR 评估:
    - 只检查 model.generate 产生的 **新 token**
    - S1->S3 必须经过 S2
    - 逐边合法性校验
    """
    model.eval()
    S1, S2, S3 = stages

    # 读取 test.txt
    by_type = {'S1->S2': [], 'S2->S3': [], 'S1->S3': []}
    with open(test_file) as f:
        for ln in f:
            toks = ln.strip().split()
            if len(toks) < 2: continue
            s, t = map(int, toks[:2])
            if   s in S1 and t in S2: by_type['S1->S2'].append((s,t))
            elif s in S2 and t in S3: by_type['S2->S3'].append((s,t))
            elif s in S1 and t in S3: by_type['S1->S3'].append((s,t))

    def decode(ids, skip):
        out = []
        for tid in ids[skip:]:
            if tid == 1: break          # newline
            if tid > 1:
                ch = itos[tid]
                if ch.isdigit(): out.append(int(ch))
        return out

    stats = {}
    for tp, lst in by_type.items():
        total = min(len(lst), max_eval_per_type)
        correct = 0
        for s, t in random.sample(lst, total):
            prompt = f"{s} {t}"
            ids = [stoi[ch] for ch in prompt if ch in stoi]
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

            y = model.generate(x, max_new_tokens=30,
                               temperature=temperature, top_k=top_k)[0]
            path = decode(y.tolist(), len(ids))

            ok = len(path) >= 1 and path[0] == s and path[-1] == t
            if ok and tp == 'S1->S3':
                ok &= any(v in S2 for v in path[1:-1])
            if ok:
                ok &= all(graph.has_edge(str(path[i]), str(path[i+1]))
                          for i in range(len(path)-1))
            correct += ok
        stats[tp] = correct/total if total else 0.0

    model.train()
    return stats
# ---------------------------------------------------------------------------