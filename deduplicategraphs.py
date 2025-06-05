# deduplicategraphs.py
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch

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

def compute_all_graph_preds(graph_list, model, device, batch_size=128):
    """
    For a list of PyG graph objects `graph_list`, run `model` on ALL of them in
    batched fashion. Return two things:
      - `all_probs`: a single NumPy array of shape [N_graphs, num_classes]
      - `all_labels`: a single NumPy array of shape [N_graphs] (the ground‐truth label)
    
    Also attach `.probs` (Tensor) to each graph object in graph_list
    """
    model.eval()
    for i, g in enumerate(graph_list):
        g.orig_index = i
    exclude = ['mc_pid', 'mc_tid', 'removal_counts']  
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, exclude_keys=exclude)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)                   # shape = [batch_size, num_classes]
            probs   = F.softmax(outputs, dim=1)      # still on device

            # Detach to CPU & convert to NumPy after stacking
            all_probs.append(probs.cpu())
            all_labels.append(batch.label.cpu())

    # Concatenate to shape [N_total, num_classes] and [N_total]
    all_probs = torch.cat(all_probs, dim=0)     # Tensor shape [N_total, C]
    all_labels = torch.cat(all_labels, dim=0)     # Tensor shape [N_total]

    # Attach probs back to each graph object in graph_list (in the same order)
    # Because DataLoader preserves order when shuffle=False, row i of all_probs
    # corresponds exactly to graph_list[i].
    for i, g in enumerate(graph_list):
        # Convert tensor row [C] → float Tensor on CPU 
        g.probs = all_probs[i].cpu().unsqueeze(0)  # shape [1, C]
        # STORE the predicted label & confidence on the graph
        conf, pred = torch.max(all_probs[i], dim=0)
        g.pred_confidence = conf.item()
        g.pred_label      = pred.item()

    return all_probs.numpy(), all_labels.numpy()

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

