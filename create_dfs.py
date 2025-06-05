# create_dfs.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import pandas as pd

def create_df_from_graphs(graph_list):

    data = []
    for graph in graph_list:
        # Extract the raw graph attributes (assumed to be a 1x24 tensor).
        raw_attr = graph.raw_graph_attr.squeeze(0).tolist()
        turning_angles = raw_attr[0:4]  # Four turning angles
        total_turn_angle    = raw_attr[4]
        chord_length_val    = raw_attr[5]
        path_length_val     = raw_attr[6]
        straightness_ratio_val = raw_attr[7]
        avg_step_length_val = raw_attr[8]
        signed_area_val     = raw_attr[9]

        # Extract local edge attributes, if present.
        if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
            local_angles      = graph.edge_attr[:, 0].tolist()
            norm_distances    = graph.edge_attr[:, 1].tolist()
            norm_lambdas      = graph.edge_attr[:, 2].tolist()
            norm_t_distances  = graph.edge_attr[:, 3].tolist()
            norm_z_distances  = graph.edge_attr[:, 4].tolist()
        else:
            local_angles = []
            norm_distances = []
            norm_lambdas = []
            norm_t_distances = []
            norm_z_distances = []

        # Get the true label.
        true_label = graph.label.item() if isinstance(graph.label, torch.Tensor) else graph.label
        correct = 1 if graph.pred_label == true_label else 0

        # For real graphs (labels 0 or 1), get the mc_pt value.
        mc_pt_val = graph.mc_pt.item()   if hasattr(graph, 'mc_pt')   else None
        mc_p_val  = graph.mc_p.item()    if hasattr(graph, 'mc_p')    else None
        mc_type_val = graph.mc_type.item() if hasattr(graph, 'mc_type') else None

        # ─── New: extract the three MC angles ───────────────────────────
        mc_phi_val   = graph.mc_phi.item()   if hasattr(graph, 'mc_phi')   else None
        mc_theta_val = graph.mc_theta.item() if hasattr(graph, 'mc_theta') else None
        mc_lam_val   = graph.mc_lam.item()   if hasattr(graph, 'mc_lam')   else None

        data.append({
            'frameId': graph.frameId,
            'label': true_label,
            'pred_label': graph.pred_label,
            'pred_confidence': graph.pred_confidence,
            'turning_angles': turning_angles,  # updated graph attributes
            'total_turn_angle': total_turn_angle,
            'chord_length': chord_length_val,
            'path_length': path_length_val,
            'straightness_ratio': straightness_ratio_val,
            'avg_step_length': avg_step_length_val,
            'signed_area': signed_area_val,
            'local_angles': local_angles,
            'norm_distances': norm_distances,
            'norm_lambdas': norm_lambdas,
            'norm_t_distances': norm_t_distances,
            'norm_z_distances': norm_z_distances,
            'mc_pt': mc_pt_val,
            'mc_p': mc_p_val,
            'mc_type': mc_type_val,
            'mc_phi': mc_phi_val,
            'mc_theta': mc_theta_val,
            'mc_lam': mc_lam_val,
            'correct': correct
        })

    df = pd.DataFrame(data)

    # Define "real" as positive (1) and "fake" as negative (0).
    df['binary_true'] = (df['label'] != 2).astype(int)
    df['binary_pred'] = (df['pred_label'] != 2).astype(int)

    df['TP'] = ((df['binary_true'] == 1) & (df['binary_pred'] == 1)).astype(int)
    df['FN'] = ((df['binary_true'] == 1) & (df['binary_pred'] == 0)).astype(int)
    df['FP'] = ((df['binary_true'] == 0) & (df['binary_pred'] == 1)).astype(int)
    df['TN'] = ((df['binary_true'] == 0) & (df['binary_pred'] == 0)).astype(int)

    return df
