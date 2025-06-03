# deduplicategraphs.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# only needs to be done for real graphs with labels 0 and 1
def compute_real_graph_preds(graphs, model, device):
    model.eval()
    with torch.no_grad():
        # First, pick out only the “real” graphs (label 0 or 1, and a valid mc_tid).
        real_graphs = [g for g in graphs if g.label.item() in [0,1] and g.mc_tid not in [None, 0]]
        for g in real_graphs:
            g_batch = Batch.from_data_list([g]).to(device)
            outputs = model(g_batch)             # shape = [1, num_classes]
            probs   = F.softmax(outputs, dim=1).squeeze(0)
            conf, pred = torch.max(probs, dim=0)
            g.pred_confidence = conf.item()
            g.pred_label      = pred.item()

def deduplicate_real_graphs(graphs):
    dedup = {}
    for g in graphs:
        # Only consider real graphs (label 0 or 1) with a valid mc_tid
        if g.label.item() in [0, 1] and g.mc_tid not in [0, None]:
            tid = g.mc_tid  # use mc_tid directly as it is stored in the graph
            # If a graph for this mc_tid already exists, compare confidences.
            if tid in dedup:
                if hasattr(g, 'pred_confidence') and hasattr(dedup[tid], 'pred_confidence'):
                    if g.pred_confidence > dedup[tid].pred_confidence:
                        dedup[tid] = g
                # Otherwise, keep the first encountered
            else:
                dedup[tid] = g
    # For fake graphs or graphs without a valid mc_tid, add them as they are.
    others = [g for g in graphs if not (g.label.item() in [0, 1] and g.mc_tid not in [0, None])]
    dedup_list = list(dedup.values())
    return dedup_list + others

