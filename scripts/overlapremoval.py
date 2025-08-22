# overlapremoval.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

def perform_overlap_removal(graph_list, dataset_name=""):
    removed_true_real = []
    removed_true_fake = []
    print(f"\n=== Performing Overlap Removal for Dataset: {dataset_name} ===")
    # Reset removal_counts on all graphs to start fresh.
    for graph in graph_list:
        if hasattr(graph, 'removal_counts'):
            del graph.removal_counts

    removal_log = { # needs initialising each time
        "original":  {"real": 0, "fake": 0},
        "predicted": {"real": 0, "fake": 0},
        "removed":   {
            "predicted_real": 0,
            "true_real": 0,
            "true_fake": 0
        },
        "final": {"real": 0, "fake": 0}
    }
    removal_confidence_diffs = []  # Initialize list for confidence differences

    # 1) Count original classes based on true labels
    for graph in graph_list:
        if graph.label.item() in [0, 1]:
            removal_log["original"]["real"] += 1
        else:
            removal_log["original"]["fake"] += 1

    # Count original classes based on true labels
    for graph in graph_list:
        if graph.label.item() in [0, 1]:
            removal_log["original"]["real"] += 1
        else:
            removal_log["original"]["fake"] += 1

    # Partition into predicted real or predicted fake
    real_graphs = [g for g in graph_list if g.pred_label in [0, 1]]
    fake_graphs = [g for g in graph_list if g.pred_label == 2]

    removal_log["predicted"]["real"] = len(real_graphs)
    removal_log["predicted"]["fake"] = len(fake_graphs)
    print(f"Initial predicted real graphs count: {len(real_graphs)}")

    # Group predicted real graphs by frameId and remove overlaps
    frames = {}
    for graph in real_graphs:
        frames.setdefault(graph.frameId, []).append(graph)

    # Overlap removal for predicted real graphs with per-survivor removal tracking
    final_real_graphs = []

    for frame_id, graphs_in_frame in frames.items():
        used_hits = []  # Reset per frame to avoid cross-frame contamination
        graphs_in_frame.sort(key=lambda g: g.pred_confidence, reverse=True)
        selected_graphs = []
        
        for graph in graphs_in_frame:
            # Convert hit positions to a set of tuples for set operations
            graph_hits = set(tuple(hit) if isinstance(hit, list) else hit for hit in graph.hit_pos)
            overlapping_candidates = []
            
            # Check against already-selected candidates in this frame
            for candidate in used_hits:
                candidate_hits, candidate_conf, candidate_ref = candidate
                if graph_hits & candidate_hits:  # They share at least one hit
                    overlapping_candidates.append(candidate)
                    
            if overlapping_candidates:
                # Get the candidate with the highest confidence among those that overlap
                best_candidate = max(overlapping_candidates, key=lambda x: x[1])
                
                # Initialize a removal counter on the surviving candidate if not already set
                if not hasattr(best_candidate[2], 'removal_counts'):
                    best_candidate[2].removal_counts = {'true_real': 0, 'true_fake': 0}
                
                # Update the counter on the surviving candidate based on the removed graph’s true label.
                if graph.label.item() in [0, 1]:
                    best_candidate[2].removal_counts['true_real'] += 1
                    removed_true_real.append(graph)
                else:
                    best_candidate[2].removal_counts['true_fake'] += 1
                    removed_true_fake.append(graph)
                    
                # Also update global removal log
                removal_log["removed"]["predicted_real"] += 1
                if graph.label.item() in [0, 1]:
                    removal_log["removed"]["true_real"] += 1
                else:
                    removal_log["removed"]["true_fake"] += 1
                    
                # Print the removal log only when a fake graph (surviving candidate is fake)
                # is removing a real graph (graph being removed is real).
                if best_candidate[2].label.item() not in [0, 1] and graph.label.item() in [0, 1]:
                    print(f"Frame {frame_id} | Removing graph with mc_tid {graph.mc_tid}, pred_confidence {graph.pred_confidence:.4f} "
                          f"because it overlaps with graph with mc_tid {best_candidate[2].mc_tid}, pred_confidence {best_candidate[1]:.4f}")

                # Record confidence difference for diagnostics
                removal_confidence_diffs.append(best_candidate[1] - graph.pred_confidence)
                # Graph is removed (not added to selected_graphs)
            else:
                # No overlap: accept the graph and record its hit set and info
                selected_graphs.append(graph)
                used_hits.append((graph_hits, graph.pred_confidence, graph))
        
        final_real_graphs.extend(selected_graphs)

    # Group surviving candidates by their true label
    removals_by_survivor_type = {
        'surviving_real': {'removed_true_real': 0, 'removed_true_fake': 0},
        'surviving_fake': {'removed_true_real': 0, 'removed_true_fake': 0}
    }

    for graph in final_real_graphs:
        if hasattr(graph, 'removal_counts'):
            if graph.label.item() in [0, 1]:
                removals_by_survivor_type['surviving_real']['removed_true_real'] += graph.removal_counts.get('true_real', 0)
                removals_by_survivor_type['surviving_real']['removed_true_fake'] += graph.removal_counts.get('true_fake', 0)
            else:  # Surviving candidate is actually fake (misclassified as real)
                removals_by_survivor_type['surviving_fake']['removed_true_real'] += graph.removal_counts.get('true_real', 0)
                removals_by_survivor_type['surviving_fake']['removed_true_fake'] += graph.removal_counts.get('true_fake', 0)

    print("Removed candidates attributed to surviving REAL graphs:")
    print("   Removed true real graphs:", removals_by_survivor_type['surviving_real']['removed_true_real'])
    print("   Removed true fake graphs:", removals_by_survivor_type['surviving_real']['removed_true_fake'])
    print("Removed candidates attributed to surviving FAKE graphs (predicted as real):")
    print("   Removed true real graphs:", removals_by_survivor_type['surviving_fake']['removed_true_real'])
    print("   Removed true fake graphs:", removals_by_survivor_type['surviving_fake']['removed_true_fake'])

    # 5) Combine surviving real graphs with all fake graphs
    final_graphs = final_real_graphs + fake_graphs

    return final_graphs, removal_log, removal_confidence_diffs, removed_true_real, removed_true_fake

