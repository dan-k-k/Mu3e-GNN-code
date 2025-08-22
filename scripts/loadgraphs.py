# loadgraphs.py
import pandas as pd

def force_label_long(graphs):
    for g in graphs:
        g.label = g.label.long()
    return graphs

def extract_unique_real(graphs):
    """
    From a list of PyG Data objects, return a DataFrame of all unique
    (mc_tid, frameId) pairs for which label ∈ {0,1}.
    """
    records = []
    for g in graphs:
        lbl = int(g.label.item())
        if lbl in (0, 1) and hasattr(g, "mc_tid") and hasattr(g, "frameId"):
            tid = g.mc_tid
            # Normalize tid → tuple
            if isinstance(tid, list):
                tid = tuple(tid)
            elif isinstance(tid, int):
                tid = (tid,)
            records.append({
                "mc_tid": tid,
                "frameId": int(g.frameId)
            })
    df = pd.DataFrame(records).drop_duplicates().sort_values(by="mc_tid")
    return df

def save_unique_real(graphs, csv_path):
    df = extract_unique_real(graphs)
    df.to_csv(csv_path, index=False)
    
