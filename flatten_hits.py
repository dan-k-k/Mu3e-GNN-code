# flatten_hits.py
# Finds the (unique) hits from all triplets. This version is used for 6hit graphs.
# iterates over all triplets to find all hits (stored as x00, x10, x20, y00, y10, y20, z00, z10, z20)
# all information is saved for each hit
# checks whether x00 is a list or a single value as the list length determines the number of triplets.
# hit ids are built from (mc_tid, triplet index, hit-within-triplet index, row index), and are useful for  
#   graph generation, ensuring the same hit is never used twice.

import uproot
import pandas as pd
import numpy as np

# Function to flatten the DataFrame
def flatten_hits(df):
    # Initialize a list to hold individual hits
    individual_hits = []

    # Iterate over each row (triplet) in the DataFrame
    for idx, row in df.iterrows():
        frame_id = row['frameId']
        mc_tid = row['mc_tid']
        mc_pid = row['mc_pid']
        mc_type = row['mc_type']
        mc_p = row['mc_p']
        mc_pt = row['mc_pt']

        # Determine the number of triplets in this row
        # Assuming all triplet coordinate lists have the same length
        n_triplets = len(row['x00']) if isinstance(row['x00'], (list, np.ndarray)) else 1

        for i in range(n_triplets):
            # Extract coordinates for the current triplet
            try:
                x = row['x00'][i] if isinstance(row['x00'], (list, np.ndarray)) else row['x00']
                y = row['y00'][i] if isinstance(row['y00'], (list, np.ndarray)) else row['y00']
                z = row['z00'][i] if isinstance(row['z00'], (list, np.ndarray)) else row['z00']

                x1 = row['x10'][i] if isinstance(row['x10'], (list, np.ndarray)) else row['x10']
                y1 = row['y10'][i] if isinstance(row['y10'], (list, np.ndarray)) else row['y10']
                z1 = row['z10'][i] if isinstance(row['z10'], (list, np.ndarray)) else row['z10']

                x2 = row['x20'][i] if isinstance(row['x20'], (list, np.ndarray)) else row['x20']
                y2 = row['y20'][i] if isinstance(row['y20'], (list, np.ndarray)) else row['y20']
                z2 = row['z20'][i] if isinstance(row['z20'], (list, np.ndarray)) else row['z20']

                # List of all hits in the triplet
                triplet_hits = [
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_0_{idx}", 'x': x, 'y': y, 'z': z},
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_1_{idx}", 'x': x1, 'y': y1, 'z': z1},
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_2_{idx}", 'x': x2, 'y': y2, 'z': z2}
                ]

                # Append each hit to the list
                for hit in triplet_hits:
                    # Ensure that x, y, z are not missing
                    if pd.notnull(hit['x']) and pd.notnull(hit['y']) and pd.notnull(hit['z']):
                        individual_hits.append(hit)
            except IndexError:
                # Skip incomplete triplets
                continue

    # Create a DataFrame for individual hits
    flat_hits_df = pd.DataFrame(individual_hits)
    return flat_hits_df

# Finds the (unique) hits from all triplets. FLAG the first hit! not used for 6hit graphs
# useful for 8hit graphs, but the following code is not used now.
def flatten_hits_with_first_flag(df, first_hit_lookup):
    individual_hits = []
    # Iterate over each row (triplet) in the DataFrame.
    for idx, row in df.iterrows():
        frame_id = row['frameId']
        mc_tid = row['mc_tid']
        mc_pid = row['mc_pid']
        mc_type = row['mc_type']
        mc_p = row['mc_p']
        mc_pt = row['mc_pt']

        n_triplets = len(row['x00']) if isinstance(row['x00'], (list, np.ndarray)) else 1

        for i in range(n_triplets):
            try:
                # Extract coordinates for the current hit in the triplet.
                x = row['x00'][i] if isinstance(row['x00'], (list, np.ndarray)) else row['x00']
                y = row['y00'][i] if isinstance(row['y00'], (list, np.ndarray)) else row['y00']
                z = row['z00'][i] if isinstance(row['z00'], (list, np.ndarray)) else row['z00']

                x1 = row['x10'][i] if isinstance(row['x10'], (list, np.ndarray)) else row['x10']
                y1 = row['y10'][i] if isinstance(row['y10'], (list, np.ndarray)) else row['y10']
                z1 = row['z10'][i] if isinstance(row['z10'], (list, np.ndarray)) else row['z10']

                x2 = row['x20'][i] if isinstance(row['x20'], (list, np.ndarray)) else row['x20']
                y2 = row['y20'][i] if isinstance(row['y20'], (list, np.ndarray)) else row['y20']
                z2 = row['z20'][i] if isinstance(row['z20'], (list, np.ndarray)) else row['z20']

                triplet_hits = [
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_0_{idx}", 'x': x, 'y': y, 'z': z},
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_1_{idx}", 'x': x1, 'y': y1, 'z': z1},
                    {'frameId': frame_id, 'mc_tid': mc_tid, 'mc_pid': mc_pid, 'mc_type': mc_type,
                     'mc_p': mc_p, 'mc_pt': mc_pt, 'mc_phi':row['mc_phi'], 'mc_theta':row['mc_theta'], 'mc_lam': row['mc_lam'],
                     'hit_id': f"{mc_tid}_{i}_2_{idx}", 'x': x2, 'y': y2, 'z': z2}
                ]

                for hit in triplet_hits:
                    if pd.notnull(hit['x']) and pd.notnull(hit['y']) and pd.notnull(hit['z']):
                        # Round hit coordinates to match the lookup precision.
                        x_r = round(hit['x'], 5)
                        y_r = round(hit['y'], 5)
                        z_r = round(hit['z'], 5)
                        # Check if the hit is a first hit based on frameId.
                        hit['first_hit'] = (frame_id in first_hit_lookup and 
                                            (x_r, y_r, z_r) in first_hit_lookup[frame_id])
                        individual_hits.append(hit)
            except IndexError:
                continue

    flat_hits_df = pd.DataFrame(individual_hits)
    return flat_hits_df

def dedupe_hits(df):
    df = df.copy()
    df['useful']  = df['mc_tid'] != 0
    for c in ('x','y','z'):
        df[f"{c}_round"] = df[c].round(5)
    df = df.sort_values('useful', ascending=False)
    grp = ['frameId','x_round','y_round','z_round']

    # 1) keep exactly one “useful” hit per (frameId, x_round, y_round, z_round)
    useful = df[df['useful']].drop_duplicates(subset=grp, keep='first')

    # 2) the remaining “non‐useful”
    non_useful = df[~df['useful']]

    # 3) find non_useful hits that do NOT share (frameId, x_round, y_round, z_round) with any useful hit
    dup_filt = (
        non_useful
        .merge(useful[grp].drop_duplicates(), on=grp, how='left', indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns=['_merge'])
    )

    # 4) now drop duplicates among those non‐useful‐without‐useful (one per group again)
    non_useful_unique = dup_filt.drop_duplicates(subset=grp, keep='first')

    # 5) concatenate “useful” + “non_useful_unique”
    result = pd.concat([useful, non_useful_unique], ignore_index=True)

    # 6) clean up the helper columns
    return result.drop(columns=['useful','x_round','y_round','z_round'])

# Assign layers to the hits
# information is based on Table 7.1 in the TDR. 
# i'm unsure exactly if there are gaps between layers along z
# and may have led to problems when deciding if a hit is forward  
# recurling when finding true and reco tids for the final efficiency comparison.

def assign_layer(row):
    x, y, z = row['x'], row['y'], row['z']
    r = np.sqrt(x**2 + y**2)
    
    # Define layer boundaries 
    if 0 < r < 29.5:
        return '1'  # Layer 1
    elif 29.5 <= r < 70:
        return '2'  # Layer 2
    elif 70 <= r < 83.3:
        if z > 351.9 / 2:
            return '3+'
        elif z < -351.9 / 2:
            return '3-'
        else:
            return '3'
    elif 83.3 <= r:
        if z > 372.6 / 2:
            return '4+'
        elif z < -372.6 / 2:
            return '4-'
        else:
            return '4'
    else:
        return '-1'  # Outside defined layers

def apply_layer_assignment(df):
    """
    Take a DataFrame of hits (with columns ['x','y','z', …]) 
    and add a 'layer' column using assign_layer.  Drop any rows
    that get a layer of '-1'.  Returns a new DataFrame with only valid layers.
    """
    # We assume df has columns 'x','y','z' (floats).
    df = df.copy()
    df['layer'] = df.apply(assign_layer, axis=1)
    df = df[df['layer'] != '-1'].reset_index(drop=True)
    return df