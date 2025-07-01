# src/create_graph.py --------------------------------------------------
import networkx as nx, random, argparse, os, pickle

def gen_graph(n_per_stage=30, p_within=0.1, p_between=0.3, seed=42):
    random.seed(seed)
    G = nx.DiGraph()
    stages = [list(range(i*n_per_stage, (i+1)*n_per_stage)) for i in range(3)]
    for s in stages: G.add_nodes_from(s)
    # intra edges
    for st in stages:
        for u in st:
            for v in st:
                if u<v and random.random()<p_within: G.add_edge(u,v)
    # inter edges S_i -> S_{i+1}
    for i in range(2):
        for u in stages[i]:
            for v in stages[i+1]:
                if random.random()<p_between: G.add_edge(u,v)
    return G, stages

def rw(G, s, t):
    try: return nx.shortest_path(G,s,t)
    except: return None

def build_sets(G, stages, k=10):
    S1,S2,S3=stages
    train, test = [], []
    for u in S1:
        for v in S2:
            if nx.has_path(G,u,v):
                for _ in range(k): train.append([u,v]+rw(G,u,v))
    for u in S2:
        for v in S3:
            if nx.has_path(G,u,v):
                for _ in range(k): train.append([u,v]+rw(G,u,v))
    for u in S1:
        for v in S3:
            p=rw(G,u,v)
            if p and any(x in S2 for x in p[1:-1]):
                test.append([u,v]+p)
    return train,test

def fmt(x): return f"{x[0]} {x[1]} "+" ".join(map(str,x[2:]))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",default="data/simple_graph/composition_90")
    args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    G,st=gen_graph()
    tr,te=build_sets(G,st,10)
    open(f"{args.out_dir}/train.txt","w").write("\n".join(map(fmt,tr)))
    open(f"{args.out_dir}/val.txt","w").write("\n".join(map(fmt,tr[:len(tr)//6])))
    open(f"{args.out_dir}/test.txt","w").write("\n".join(map(fmt,te)))
    nx.write_graphml(G,f"{args.out_dir}/graph.graphml")
    pickle.dump({'stages':st},open(f"{args.out_dir}/stage.pkl","wb"))
    print("Graph & txt written to",args.out_dir)
# ---------------------------------------------------------------------