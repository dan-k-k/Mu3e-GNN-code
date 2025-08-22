# truetrackbuilding.py

import numpy as np
import pandas as pd

def explode_triplets_to_tracks(segs_df):
    """
    Given a DataFrame `segs_df` whose rows each contain:
      - 'x00','y00','z00','x10','y10','z10','x20','y20','z20'
      - plus the MC labels: 'frameId', 'mc_tid', 'mc_pid', 'mc_type', 'mc_p', 'mc_pt'
    this function “explodes” each row into its constituent triplets and then
    stitches those triplets into 3-hit (or more) tracks. Returns a DataFrame
    tracks_df with columns:
      ['frameId','mc_tid','mc_pid','mc_type','mc_p','mc_pt','hits']
    where each row’s 'hits' is a Python list of (x,y,z) tuples in the correct order.
    """
    exploded_rows = []

    # 1) EXPLODE INTO “ONE-TRIPLET-PER-ROW”
    for idx, row in segs_df.iterrows():
        frame_id = row['frameId']
        mc_tid   = row['mc_tid']
        mc_pid   = row['mc_pid']
        mc_type  = row['mc_type']
        mc_p     = row['mc_p']
        mc_pt    = row['mc_pt']

        # How many triplets live in this row?  If 'x00' is a list, length = number of triplets.
        n_triplets = len(row['x00']) if isinstance(row['x00'], (list, np.ndarray)) else 1

        for i in range(n_triplets):
            try:
                # Grab the 3 hits of this one triplet (round to 5 decimals)
                h0 = (round(row['x00'][i], 5), round(row['y00'][i], 5), round(row['z00'][i], 5))
                h1 = (round(row['x10'][i], 5), round(row['y10'][i], 5), round(row['z10'][i], 5))
                h2 = (round(row['x20'][i], 5), round(row['y20'][i], 5), round(row['z20'][i], 5))

                exploded_rows.append({
                    'frameId': frame_id,
                    'mc_tid': mc_tid,
                    'mc_pid': mc_pid,
                    'mc_type': mc_type,
                    'mc_p': mc_p,
                    'mc_pt': mc_pt,
                    'triplet': (h0, h1, h2)
                })
            except (IndexError, TypeError):
                # Skip if any coordinate is missing or lists are too short
                continue

    exploded_df = pd.DataFrame(exploded_rows)
    # Deduplicate identical (frameId, mc_tid, triplet) rows
    exploded_df.drop_duplicates(subset=['frameId', 'mc_tid', 'triplet'], inplace=True)
    exploded_df.reset_index(drop=True, inplace=True)

    # 2) GROUP BY (frameId,mc_tid) AND “STITCH” TRIPLETS INTO 3+ HIT TRACKS
    tracks = []
    grouped = exploded_df.groupby(['frameId', 'mc_tid'], sort=False)

    for (frame_id, mc_tid), group in grouped:
        group = group.copy()
        triplets = group['triplet'].tolist()
        mc_p  = group['mc_p'].iloc[0]
        mc_pt = group['mc_pt'].iloc[0]
        mc_pid  = group['mc_pid'].iloc[0]
        mc_type = group['mc_type'].iloc[0]

        group_tracks = []
        for triplet in triplets:
            # Build three “hits” for this triplet
            hits = [
                triplet[0],  # (x00,y00,z00)
                triplet[1],  # (x10,y10,z10)
                triplet[2]   # (x20,y20,z20)
            ]

            # Find existing tracks to append or prepend
            matching_append = []
            matching_prepend = []
            for t in group_tracks:
                last_two  = t['hits'][-2:]
                first_two = t['hits'][:2]

                if last_two == hits[:2]:
                    matching_append.append(t)
                if first_two == hits[-2:]:
                    matching_prepend.append(t)

            matched_any = False
            # Append third‐hit to all that matched on last_two == hits[:2]
            for t in matching_append:
                t['hits'].append(hits[2])
                t['triplets'].append(triplet)
                matched_any = True

            # Prepend first‐hit to all that matched on first_two == hits[-2:]
            for t in matching_prepend:
                t['hits'].insert(0, hits[0])
                t['triplets'].insert(0, triplet)
                matched_any = True

            # If no match, start a brand‐new track
            if not matched_any:
                group_tracks.append({
                    'hits': hits.copy(),
                    'triplets': [triplet],
                    'mc_pid': mc_pid,
                    'mc_type': mc_type,
                    'mc_p': mc_p,
                    'mc_pt': mc_pt
                })

        # Finally, any “track” that has ≥3 hits becomes a row in tracks
        for t in group_tracks:
            if len(t['hits']) >= 3:
                tracks.append({
                    'frameId': frame_id,
                    'mc_tid': mc_tid,
                    'mc_pid': t['mc_pid'],
                    'mc_type': t['mc_type'],
                    'mc_p': t['mc_p'],
                    'mc_pt': t['mc_pt'],
                    'hits': t['hits']
                })

    tracks_df = pd.DataFrame(tracks)
    return tracks_df

def assign_layer1(hit):
    x, y, z = hit
    r = np.sqrt(x**2 + y**2)
    
    # Define layer boundaries (modify as per your detector's geometry)
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

