# src/prepare_data.py -----------------------------------------------
import os, pickle, argparse, numpy as np, json, re

def build_vocab(texts):
    chars=sorted(set("".join(texts)))
    stoi={'[PAD]':0,'\n':1}
    for ch in chars:
        if ch!='\n': stoi[ch]=len(stoi)
    itos={i:c for c,i in stoi.items()}
    return stoi,itos

def encode(s,stoi): return [stoi.get(ch,0) for ch in s]

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default="data/simple_graph/composition_90")
    args=ap.parse_args()
    tr=open(f"{args.dir}/train.txt").read().strip().splitlines()
    val=open(f"{args.dir}/val.txt").read().strip().splitlines()
    te=open(f"{args.dir}/test.txt").read().strip().splitlines()
    stoi,itos=build_vocab(tr+val+te)
    blk=((max(map(len,tr))+5)*3+10+31)//32*32
    meta={'stoi':stoi,'itos':itos,'vocab_size':len(stoi),'block_size':blk}
    pickle.dump(meta,open(f"{args.dir}/meta.pkl","wb"))
    for split,name in [(tr,'train'),(val,'val')]:
        tokens=[]
        for ln in split:
            ids=encode(ln+'\n',stoi)
            ids=ids[:blk+1]+[0]*(blk+1-len(ids))
            tokens.extend(ids)
        np.array(tokens,dtype=np.uint16).tofile(f"{args.dir}/{name}.bin")
    print("meta & bin saved")
# --------------------------------------------------------------------