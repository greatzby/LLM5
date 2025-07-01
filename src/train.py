#!/usr/bin/env python
# ---------------------------------------------------------
# LLM5 / src/train.py
# 训练 3-stage 组合路径模型 + 在线评测
# 默认 1-layer 1-head 120-embd，可通过 CLI 修改
# ---------------------------------------------------------
import os, argparse, pickle, random, numpy as np, torch, networkx as nx
from datetime import datetime
from model import GPTConfig, GPT
from evaluate import evaluate_ar          # 需确保 src/evaluate.py 已放好

# ---------- 简易 Logger ----------
def get_logger(path):
    class _L:
        def __init__(self,p): self.f=open(p,'w')
        def info(self,msg): print(msg); self.f.write(msg+'\n'); self.f.flush()
    return _L(path)
# ----------------------------------

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/simple_graph/composition_90')
    p.add_argument('--n_layer', type=int, default=1)
    p.add_argument('--n_head',  type=int, default=1)
    p.add_argument('--n_embd',  type=int, default=120)
    p.add_argument('--max_iters', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--betas', type=float, nargs=2, default=(0.9,0.95))
    p.add_argument('--device', default='cuda')
    p.add_argument('--test_interval', type=int, default=1000)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()
# -------------------------

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --------- batch util (修正 uint16) ----------
def get_batch(arr, bs, block_size, device):
    N = len(arr)//(block_size+1)
    idx = torch.randint(0, N, (bs,))
    idx = idx*(block_size+1)
    x = torch.stack([torch.from_numpy(arr[i:i+block_size].astype(np.int64))
                     for i in idx]).to(device)
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+block_size].astype(np.int64))
                     for i in idx]).to(device)
    return x.long(), y.long()
# --------------------------------------------

def main():
    args = parse_args(); seed_all(args.seed)

    out_dir = f"out/run_{datetime.now().strftime('%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "train.log"))

    # -------- 数据加载 --------
    meta = pickle.load(open(f"{args.data_dir}/meta.pkl", 'rb'))
    block_size = meta['block_size']; vocab_size = meta['vocab_size']
    stoi, itos = meta['stoi'], meta['itos']

    train_bin = np.memmap(f"{args.data_dir}/train.bin", dtype=np.uint16, mode='r')
    val_bin   = np.memmap(f"{args.data_dir}/val.bin",   dtype=np.uint16, mode='r')
    stages = pickle.load(open(f"{args.data_dir}/stage.pkl", 'rb'))['stages']
    G = nx.read_graphml(f"{args.data_dir}/graph.graphml")

    print(f"Data loaded. train seq={len(train_bin)//(block_size+1)}, "
          f"val seq={len(val_bin)//(block_size+1)}")

    # -------- 模型 --------
    cfg = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, dropout=0.0, bias=False)
    model = GPT(cfg).to(args.device)

    opt = model.configure_optimizers(weight_decay=args.weight_decay,
                                     learning_rate=args.lr,
                                     betas=tuple(args.betas),
                                     device_type='cuda' if 'cuda' in args.device else 'cpu')

    # -------- 训练循环 --------
    running_loss = 0.0; n_loss = 0
    for it in range(args.max_iters+1):

        # ----- 定期评估 -----
        if it % args.test_interval == 0:
            model.eval()
            with torch.no_grad():
                vloss = np.mean([ model(*get_batch(val_bin,256,block_size,args.device))[1].item()
                                  for _ in range(8) ])
            ar = evaluate_ar(model, f"{args.data_dir}/test.txt", stages,
                             stoi, itos, args.device, G,
                             temperature=args.temperature, top_k=args.top_k)
            tloss = running_loss / n_loss if n_loss else 0.0
            logger.info(f"{it:6d} | train {tloss:.4f} | val {vloss:.4f} | "
                        f"S1S2 {ar['S1->S2']:.2%} | S2S3 {ar['S2->S3']:.2%} | "
                        f"S1S3 {ar['S1->S3']:.2%}")
            running_loss = 0.0; n_loss = 0
            model.train()

        # ----- 终止 -----
        if it == args.max_iters: break

        # ----- 训练一步 -----
        xb,yb = get_batch(train_bin, args.batch_size, block_size, args.device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running_loss += loss.item(); n_loss += 1

if __name__ == "__main__":
    main()