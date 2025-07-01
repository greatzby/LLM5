# src/train.py --------------------------------------------------------
import os, argparse, pickle, torch, numpy as np, networkx as nx
from datetime import datetime
from model import GPTConfig,GPT
from evaluate import evaluate_ar
from logger import get_logger

def arg():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',default='data/simple_graph/composition_90')
    p.add_argument('--n_layer',type=int,default=2)
    p.add_argument('--n_head',type=int,default=4)
    p.add_argument('--n_embd',type=int,default=256)
    p.add_argument('--max_iters',type=int,default=20000)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--device',default='cuda')
    p.add_argument('--lr',type=float,default=3e-4)
    return p.parse_args()

def get_batch(arr,bs,blk,dev):
    N=len(arr)//(blk+1)
    ix=torch.randint(0,N,(bs,))
    ix=ix*(blk+1)
    xs=np.stack([arr[i:i+blk] for i in ix])
    ys=np.stack([arr[i+1:i+1+blk] for i in ix])
    return torch.tensor(xs,device=dev),torch.tensor(ys,device=dev)

if __name__=="__main__":
    a=arg(); os.makedirs('out',exist_ok=True)
    out=f"out/run_{datetime.now().strftime('%m%d_%H%M%S')}"
    os.makedirs(out); log=get_logger(f"{out}/log.txt")

    meta=pickle.load(open(f"{a.data_dir}/meta.pkl","rb"))
    blk,vsz=meta['block_size'],meta['vocab_size']
    stoi,itos=meta['stoi'],meta['itos']
    tr=np.memmap(f"{a.data_dir}/train.bin",dtype=np.uint16,mode='r')
    va=np.memmap(f"{a.data_dir}/val.bin",dtype=np.uint16,mode='r')
    G=nx.read_graphml(f"{a.data_dir}/graph.graphml")
    stages=pickle.load(open(f"{a.data_dir}/stage.pkl","rb"))['stages']

    model=GPT(GPTConfig(vocab_size=vsz,block_size=blk,
                        n_layer=a.n_layer,n_head=a.n_head,
                        n_embd=a.n_embd)).to(a.device)
    opt=model.configure_optimizers(lr=a.lr,device_type='cuda')

    for it in range(a.max_iters+1):
        if it%1000==0:
            model.eval()
            loss=np.mean([model(*get_batch(va,256,blk,a.device))[1].item() for _ in range(10)])
            acc=evaluate_ar(model,f"{a.data_dir}/test.txt",stages,stoi,itos,a.device,G)
            print(f"{it:5d} | val_loss {loss:.3f} | S1S2 {acc['S1->S2']:.2%} "
                  f"S2S3 {acc['S2->S3']:.2%} S1S3 {acc['S1->S3']:.2%}")
            model.train()
        xb,yb=get_batch(tr,a.batch,blk,a.device)
        _,l=model(xb,yb); opt.zero_grad(); l.backward(); opt.step()
# --------------------------------------------------------------------