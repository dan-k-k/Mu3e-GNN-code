# run_inference.py

import argparse
import uproot
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Import your custom modules
from scripts.flatten_hits import flatten_hits, dedupe_hits, apply_layer_assignment
from scripts.build6hittracks import build_and_validate_tracks
from scripts.graphgeneration import track_to_graph6, layer_map
from scripts.defineGNNmodel import GCNMultiClass # Or GINEMultiClass, whichever you used
from scripts.deduplicategraphs import compute_all_graph_preds, deduplicate_real_graphs
from scripts.overlapremoval import perform_overlap_removal

def main(args):
    """Main inference pipeline function."""
    
    # 1. Setup: Device, Model, and Stats
    # ------------------------------------
    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    print("Loading normalization statistics...")
    stats = np.load(args.stats_path)
    geom_means = stats['geom_means']
    geom_stds = stats['geom_stds']
    global_distance_means = stats['distance_means']
    global_distance_stds = stats['distance_stds']
    global_lambda_means = stats['lambda_means']
    global_lambda_stds = stats['lambda_stds']
    global_t_means = stats['t_means']
    global_t_stds = stats['t_stds']
    global_z_means = stats['z_means']
    global_z_stds = stats['z_stds']

    print("Loading trained GNN model...")
    # IMPORTANT: Ensure these parameters match your trained model
    model = GCNMultiClass(in_channels=11, hidden_channels=64, extra_features=10) 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Data Loading and Preprocessing
    # ---------------------------------
    print(f"Loading data from {args.input_file}...")
    with uproot.open(args.input_file) as file:
        segs_tree = file[args.tree_name]
        segs_branches = ['x00','y00','z00','x10','y10','z10','x20','y20','z20', 'frameId']
        # Load only necessary branches for inference
        segs_df = segs_tree.arrays(segs_branches, library='pd')

    print("Flattening, deduplicating, and assigning layers to hits...")
    hits_df_flat = flatten_hits(segs_df)
    hits_df_unique = dedupe_hits(hits_df_flat)
    hits_df_layered = apply_layer_assignment(hits_df_unique)
    
    unique_frame_ids = sorted(hits_df_layered['frameId'].unique())
    print(f"Found {len(unique_frame_ids)} frames to process.")

    # 3. Frame-by-Frame Inference Loop
    # --------------------------------
    all_surviving_graphs = []
    
    # Define the layer sequences to check for tracks
    layer_sequences = [
        ('1', '2', '3', '4', '4+', '3+'),
        ('1', '2', '3', '4', '4-', '3-')
    ]

    for frame_id in tqdm(unique_frame_ids, desc="Processing Frames"):
        # A. Build Track Candidates
        validated_tracks = build_and_validate_tracks(
            hits_df_layered,
            frame_id,
            layer_sequences=layer_sequences,
            # Use default tolerances or pass them via args
            center_tolerance=50,
            radius_tolerance=50,
            pitch_tolerance=40
        )
        if not validated_tracks:
            continue

        # B. Convert Tracks to Graphs
        graph_list = []
        for track in validated_tracks:
            # For inference, the label is irrelevant. Use a placeholder like -1.
            graph = track_to_graph6(
                track, label=-1, layer_map=layer_map,
                geom_means=geom_means, geom_stds=geom_stds,
                global_distance_means=global_distance_means, global_distance_stds=global_distance_stds,
                global_lambda_means=global_lambda_means, global_lambda_stds=global_lambda_stds,
                global_t_means=global_t_means, global_t_stds=global_t_stds,
                global_z_means=global_z_means, global_z_stds=global_z_stds
            )
            graph_list.append(graph)
        
        if not graph_list:
            continue

        # C. Run GNN Prediction and Post-processing
        compute_all_graph_preds(graph_list, model, device)
        
        # NOTE: deduplicate_real_graphs requires true labels and mc_tid, which are not
        # available in pure inference. We skip it and go straight to overlap removal.
        # This is a key difference between evaluation and deployment.
        
        survivors, _, _, _, _ = perform_overlap_removal(graph_list, dataset_name=f"Frame_{frame_id}")
        all_surviving_graphs.extend(survivors)

    # 4. Save Results
    # ---------------
    print(f"\nProcessed all frames. Found {len(all_surviving_graphs)} final tracks.")
    if all_surviving_graphs:
        output_data = []
        for g in all_surviving_graphs:
            hit_ids = [hit['hit_id'] for hit in g.hits] # Assuming 'hits' is stored on the graph object
            output_data.append({
                'frameId': g.frameId,
                'predicted_label': g.pred_label,
                'confidence': g.pred_confidence,
                'num_hits': len(g.hit_pos),
                'hit_ids': ','.join(hit_ids) # Storing hit IDs as a comma-separated string
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
    else:
        print("No tracks survived the pipeline. No output file created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 6-hit track finding GNN inference pipeline.")
    parser.add_argument('--input-file', type=str, required=True, help="Path to the input ROOT file.")
    parser.add_argument('--tree-name', type=str, default='segs;10', help="Name of the TTree in the ROOT file.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained GNN model (.pt file).")
    parser.add_argument('--stats-path', type=str, required=True, help="Path to the normalization stats (.npz file).")
    parser.add_argument('--output-file', type=str, default='reconstructed_tracks.csv', help="Path to save the output CSV file.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu').")
    
    args = parser.parse_args()
    main(args)

