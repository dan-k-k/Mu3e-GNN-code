# graphcreation.py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd

# A fixed map from layer‐string → index (0 to 7). Used for one‐hot encoding. 
 
layer_map = {
    "1":  0,
    "2":  1,
    "3":  2,
    "4":  3,
    "4+": 4,
    "3+": 5,
    "4-": 6,
    "3-": 7,
}

# Edge‐and‐geometry helper functions

def compute_lambda(hit1: dict, hit2: dict) -> float:
    """
    Compute the “lambda” angle between two hits:
      λ = arctan(|Δz| / Δr)  where Δr = sqrt((Δx)^2 + (Δy)^2).
    If Δr ≈ 0 but Δz != 0, return ±π/2 accordingly. If both Δr and Δz are zero, return 0.
    """
    dx = hit2["x"] - hit1["x"]
    dy = hit2["y"] - hit1["y"]
    dr = np.hypot(dx, dy)
    dz = abs(hit2["z"] - hit1["z"])

    if dr < 1e-9:
        # vertical line in z:
        if dz > 0:
            return np.pi / 2
        elif dz < 0:
            return -np.pi / 2
        else:
            return 0.0

    return np.arctan2(dz, dr)


def compute_all_lambdas_6hit(track_hits: list) -> list:
    """
    Given a 6‐hit track (list of six hit‐dicts), return a list of length 5:
      [ λ(h0,h1), λ(h1,h2), λ(h2,h3), λ(h3,h4), λ(h4,h5) ].
    If len(track_hits) != 6, returns an empty list or partial list of None’s.
    """
    lambdas = [None] * 5
    if len(track_hits) == 6:
        lambdas[0] = compute_lambda(track_hits[0], track_hits[1])
        lambdas[1] = compute_lambda(track_hits[1], track_hits[2])
        lambdas[2] = compute_lambda(track_hits[2], track_hits[3])
        lambdas[3] = compute_lambda(track_hits[3], track_hits[4])
        lambdas[4] = compute_lambda(track_hits[4], track_hits[5])
    return lambdas


def compute_turning_angle(h_prev: dict, h_curr: dict, h_next: dict) -> float:
    """
    Compute the planar “turning angle” at h_curr given three consecutive hits:
      TA = | wrap( ϕ(h_{i+1}) – ϕ(h_i) ) |, wrapped to [–π, +π], then absolute value.
    """
    angle1 = np.arctan2(h_curr["y"] - h_prev["y"], h_curr["x"] - h_prev["x"])
    angle2 = np.arctan2(h_next["y"] - h_curr["y"], h_next["x"] - h_curr["x"])
    diff   = angle2 - angle1
    # wrap into (–π, +π):
    diff   = (diff + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def compute_all_turning_angles_6hit(track_hits: list) -> list:
    """
    Returns a list of four turning‐angles for a 6‐hit track:
      [ TA(h0,h1,h2), TA(h1,h2,h3), TA(h2,h3,h4), TA(h3,h4,h5) ].
    If len(track_hits) != 6, returns a list of None’s or empty.
    """
    angles = [None] * 4
    if len(track_hits) == 6:
        angles[0] = compute_turning_angle(track_hits[0], track_hits[1], track_hits[2])
        angles[1] = compute_turning_angle(track_hits[1], track_hits[2], track_hits[3])
        angles[2] = compute_turning_angle(track_hits[2], track_hits[3], track_hits[4])
        angles[3] = compute_turning_angle(track_hits[3], track_hits[4], track_hits[5])
    return angles


def get_layer_one_hot(hit: dict, layer_map: dict, num_layers: int = 8) -> torch.Tensor:
    """
    Given a single hit‐dict containing a 'layer' key (e.g. "1","2","3",...),
    return an 8‐dim one‐hot tensor.  E.g. if layer_map["3+"] == 5, then output
    F.one_hot(torch.tensor(5), num_classes=8).float().
    """
    layer_idx = layer_map[hit["layer"]]
    return F.one_hot(torch.tensor(layer_idx), num_classes=num_layers).float()


def compute_total_turning_angle(hits: list) -> float:
    """
    Compute the sum of all planar-angles between successive hits, in the XY plane.
    That is, sum_over_i [ wrap(ϕ_{i+1} – ϕ_i) ], wrapped to (–π, +π) before summation.
    """
    angles = [np.arctan2(h["y"], h["x"]) for h in hits]
    angles = np.array(angles)
    diffs  = np.diff(angles)
    # wrap each difference into (–π, +π):
    diffs  = (diffs + np.pi) % (2 * np.pi) - np.pi
    return np.sum(diffs)


def chord_length(hits: list) -> float:
    """
    Euclidean distance between the first hit and the last hit in 3D.
    """
    first = hits[0]
    last  = hits[-1]
    return np.sqrt(
        (last["x"] - first["x"]) ** 2 +
        (last["y"] - first["y"]) ** 2 +
        (last["z"] - first["z"]) ** 2
    )


def path_length(hits: list) -> float:
    """
    Sum of successive 3D‐distances along the track: 
      ∑_{i=0 to n-2} sqrt((x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2 + (z_{i+1}-z_i)^2).
    """
    total = 0.0
    for i in range(len(hits) - 1):
        dx = hits[i+1]["x"] - hits[i]["x"]
        dy = hits[i+1]["y"] - hits[i]["y"]
        dz = hits[i+1]["z"] - hits[i]["z"]
        total += np.sqrt(dx*dx + dy*dy + dz*dz)
    return total


def straightness_ratio(hits: list) -> float:
    """
    Ratio = chord_length(hits) / path_length(hits).  If path_length == 0, returns 0.0.
    """
    pl = path_length(hits)
    if pl > 0:
        return chord_length(hits) / pl
    return 0.0


def average_step_length(hits: list) -> float:
    """
    path_length(hits) / (number_of_steps), where number_of_steps = len(hits) - 1.
    If len(hits) < 2, returns 0.0.
    """
    n = len(hits)
    if n > 1:
        return path_length(hits) / (n - 1)
    return 0.0


def signed_area(hits: list) -> float:
    """
    Compute the signed polygon area in the XY plane, returning
      (1/2) * ∑_{i=0..n-1} (x_i * y_{i+1} - x_{i+1} * y_i), 
    where we “close” the loop by taking (x_n, y_n) = (x_0, y_0).
    """
    xs = [h["x"] for h in hits]
    ys = [h["y"] for h in hits]
    xs.append(xs[0])
    ys.append(ys[0])
    arr_x = np.array(xs[:-1])
    arr_y = np.array(ys[:-1])
    arr_xp = np.array(xs[1:])
    arr_yp = np.array(ys[1:])
    area = 0.5 * np.sum(arr_x * arr_yp - arr_xp * arr_y)
    return area

# 3) “Constraint‐vector” extraction for one 6‐hit track

def extract_constraints6(track: dict) -> np.ndarray:
    """
    Given one “track” dict that has track['hits'] = a length‐6 list of hit‐dicts,
    compute the following features (in order):
      [ TA0, TA1, TA2, TA3, (replaced None→0.0),
        total_turning_angle,
        chord_length,
        path_length,
        straightness_ratio,
        average_step_length,
        signed_area ]
    where TAi = compute_all_turning_angles_6hit(track['hits'])[i], 
    and we replace any None by 0.0 if not enough hits.
    Returns a NumPy array of shape (10,) dtype float32.
    """
    hits = track["hits"]

    # 1) Compute the four turning angles, replace None→0.0
    turning_angles = compute_all_turning_angles_6hit(hits)
    turning_angles = [
        (ta if (ta is not None) else 0.0) for ta in turning_angles
    ]

    # 2) Now append the remaining “global” geometry features:
    total_ta = compute_total_turning_angle(hits)
    chord    = chord_length(hits)
    pl       = path_length(hits)
    straight = straightness_ratio(hits)
    avg_stp  = average_step_length(hits)
    area     = signed_area(hits)

    constraints = turning_angles + [
        total_ta,
        chord,
        pl,
        straight,
        avg_stp,
        area,
    ]
    return np.array(constraints, dtype=np.float32)


def extract_geom_features_from_df6(df: "pd.DataFrame") -> np.ndarray:
    """
    Given a DataFrame whose column 'hits' holds length‐6 lists of hit‐dicts,
    apply extract_constraints6(row) row by row. Returns a 2D array of shape (N, 10).
    """
    feat_list = []
    for _, row in df.iterrows():
        feat_list.append(extract_constraints6(row))
    return np.vstack(feat_list)

# “Global edge‐feature stats” from a list of tracks

def compute_global_edge_features_stats6(tracks: list) -> dict:
    """
    Given a list of track‐dicts (each with track['hits']), compute means+stds
    of the following per‐edge features:
      - distance   = Euclidean 3D length of each edge
      - tdist      = transverse (XY) distance (= sqrt(dx^2 + dy^2))
      - zdist      = Δz
      - lambda     = compute_lambda(hit_i, hit_{i+1})

    Returns a dict:
      {
        "distance": ( [mean_edge0, mean_edge1, …], [std_edge0, std_edge1, …] ),
        "tdist":    ( [mean_edge0, …],             [std_edge0, …] ),
        "zdist":    ( [mean_edge0, …],             [std_edge0, …] ),
        "lambda":   ( [mean_edge0, …],             [std_edge0, …] )
      }
    """
    edge_stats = {
        "distance": {},
        "tdist":    {},
        "zdist":    {},
        "lambda":   {}
    }

    # Accumulate lists of values for each edge index
    for track in tracks:
        hits = track["hits"]
        n    = len(hits)
        if n < 2:
            continue
        for i in range(n - 1):
            h1 = hits[i]
            h2 = hits[i + 1]
            dx = h2["x"] - h1["x"]
            dy = h2["y"] - h1["y"]
            dz = h2["z"] - h1["z"]

            dist  = np.sqrt(dx*dx + dy*dy + dz*dz)
            tdist = np.sqrt(dx*dx + dy*dy)
            lam   = compute_lambda(h1, h2)

            for feat_name, val in [
                ("distance", dist),
                ("tdist",    tdist),
                ("zdist",    dz),
                ("lambda",   (0.0 if (lam is None or np.isnan(lam)) else lam))
            ]:
                if i not in edge_stats[feat_name]:
                    edge_stats[feat_name][i] = []
                edge_stats[feat_name][i].append(val)

    # Compute mean+std for each edge‐index, for each feature
    results = {}
    for feat_name, per_edge_dict in edge_stats.items():
        means = []
        stds  = []
        max_edge = max(per_edge_dict.keys()) if per_edge_dict else -1
        for edge_i in range(max_edge + 1):
            values = per_edge_dict.get(edge_i, [])
            if len(values) > 0:
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            else:
                means.append(0.0)
                stds.append(1.0)
        results[feat_name] = (means, stds)

    return results

# 5) Convert a single 6‐hit “track” → torch_geometric.data.Data

def track_to_graph6(
    track: dict,
    label: int,
    layer_map: dict,
    geom_means: np.ndarray,
    geom_stds: np.ndarray,
    global_distance_means: list,
    global_distance_stds:  list,
    global_lambda_means:   list,
    global_lambda_stds:    list,
    global_t_means:        list,
    global_t_stds:         list,
    global_z_means:        list,
    global_z_stds:         list
) -> Data:
    """
    Given one track-dict with:
      - track['hits']: 6-element list of hit‐dicts (each has keys 'x','y','z','layer', etc.)
      - track['frameId'], track['mc_tid'], track['mc_type'], track['mc_pt'], track['mc_p'], track['mc_phi'], track['mc_theta'], track['mc_lam'], etc.

    and given:
      - label: an integer (0=e+,1=e-, 2=fake, etc.)
      - layer_map: dictionary from layer‐string → index (0..7)
      - geom_means/geom_stds:  length‐10 arrays for normalizing the 10 “constraint” features
      - global_*_means/stds:  lists of per‐edge means/stds for “distance”, “lambda”, etc.

    Returns a torch_geometric.data.Data object with:
      x, edge_index, edge_attr, graph_attr (normalized), raw_graph_attr, label, plus metadata fields.
    """
    hits = track["hits"]
    n_hits = len(hits)

    # 1) Node features:  [ x, y, z, one-hot(layer) ] for each hit
    coords = [[h["x"], h["y"], h["z"]] for h in hits]
    x_tensor = torch.tensor(coords, dtype=torch.float)              # shape = (n_hits, 3)
    one_hots  = [get_layer_one_hot(h, layer_map) for h in hits]      # each is (8,)
    layer_feats = torch.stack(one_hots, dim=0)                       # shape = (n_hits, 8)
    x_combined  = torch.cat([x_tensor, layer_feats], dim=1)          # shape = (n_hits, 11)

    # 2) Build a “chain” edge_index between consecutive hits
    if n_hits > 1:
        num_edges = n_hits - 1
        edge_index = torch.tensor([
            list(range(num_edges)),
            list(range(1, num_edges + 1))
        ], dtype=torch.long)  # shape = (2, num_edges)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 3) For each edge i→i+1, compute (local_angle, normalized distance, normalized lambda, normalized tdist, normalized zdist)
    if n_hits > 1:
        # Convert coords to a tensor for easy diffs:
        coords_tensor = x_tensor  # shape = (n_hits, 3)
        dx = coords_tensor[1:, 0] - coords_tensor[:-1, 0]  # (num_edges,)
        dy = coords_tensor[1:, 1] - coords_tensor[:-1, 1]
        dz = coords_tensor[1:, 2] - coords_tensor[:-1, 2]

        # Local “phi‐angle” feature = atan2(dy, dx).  → shape = (num_edges,)
        local_angles = torch.atan2(dy, dx).unsqueeze(1)    # shape = (num_edges, 1)

        # Euclidean distance & transverse distance & raw lambda & raw z‐distance
        dist_raw  = torch.sqrt(dx*dx + dy*dy + dz*dz).unsqueeze(1)  # (num_edges,1)
        tdist_raw = torch.sqrt(dx*dx + dy*dy).unsqueeze(1)         # (num_edges,1)
        lam_list  = compute_all_lambdas_6hit(hits)                  # Python list of length 5 (if n_hits=6)
        lam_raw   = torch.tensor(lam_list[:num_edges], dtype=torch.float).unsqueeze(1)  # (num_edges,1)
        zdist_raw = dz.unsqueeze(1)                                 # (num_edges,1)

        # Normalize each “raw” quantity edge‐by‐edge
        # (We assume global_*_means and _stds are Python lists whose length ≥ num_edges)
        dist_norm = (dist_raw - torch.tensor(global_distance_means[:num_edges], dtype=torch.float).unsqueeze(1)) \
                     / (torch.tensor(global_distance_stds[:num_edges], dtype=torch.float).unsqueeze(1) + 1e-8)
        lam_norm  = (lam_raw  - torch.tensor(global_lambda_means[:num_edges], dtype=torch.float).unsqueeze(1)) \
                     / (torch.tensor(global_lambda_stds[:num_edges], dtype=torch.float).unsqueeze(1) + 1e-8)
        tdist_norm= (tdist_raw- torch.tensor(global_t_means[:num_edges], dtype=torch.float).unsqueeze(1)) \
                     / (torch.tensor(global_t_stds[:num_edges], dtype=torch.float).unsqueeze(1) + 1e-8)
        zdist_norm= (zdist_raw- torch.tensor(global_z_means[:num_edges], dtype=torch.float).unsqueeze(1)) \
                     / (torch.tensor(global_z_stds[:num_edges], dtype=torch.float).unsqueeze(1) + 1e-8)

        # Concatenate into a single edge_attr matrix of size (num_edges, 1+1+1+1+1) = (num_edges, 5)
        edge_attr = torch.cat([ local_angles, dist_norm, lam_norm, tdist_norm, zdist_norm ], dim=1)
    else:
        edge_attr  = torch.empty((0, 5), dtype=torch.float)

    # 4) Build “constraint vector” and normalize it
    raw_constraints      = extract_constraints6(track)      # shape = (10,)
    normalized_constraints = (raw_constraints - geom_means) / (geom_stds + 1e-8)
    graph_attr_tensor   = torch.tensor(normalized_constraints, dtype=torch.float).unsqueeze(0)  # (1,10)
    raw_graph_attr      = torch.tensor(raw_constraints, dtype=torch.float).unsqueeze(0)         # (1,10)

    # 5) Construct the Data object and attach metadata
    data = Data(
        x         = x_combined,         # shape = (n_hits, 11)
        edge_index= edge_index,         # shape = (2, num_edges)
        edge_attr = edge_attr,          # shape = (num_edges, 5)
        label = torch.tensor([label], dtype=torch.long)  # 1‐element tensor storing the graph‐label
    )
    # (Attach extra fields to data for later reference)
    data.hit_pos = [tuple(round(c, 5) for c in hit) for hit in coords]
    data.frameId    = track.get("frameId", None)
    data.mc_tid     = track.get("mc_tid", None)
    data.layers     = [h["layer"] for h in hits]
    data.raw_graph_attr = raw_graph_attr
    data.graph_attr     = graph_attr_tensor

    # Also store MC‐level features if available:
    pt  = track.get("mc_pt", None)
    p   = track.get("mc_p",  None)
    data.mc_pt    = torch.tensor(pt  if pt  is not None else float("nan"), dtype=torch.float)
    data.mc_p     = torch.tensor(p   if p   is not None else float("nan"), dtype=torch.float)

    mc_type = track.get("mc_type", None)
    if (mc_type is None) or (mc_type == "Multiple"):
        data.mc_type = torch.tensor(float("nan"), dtype=torch.float)
    else:
        data.mc_type = torch.tensor(float(mc_type), dtype=torch.float)

    data.mc_phi   = torch.tensor(track.get("mc_phi",   float("nan")), dtype=torch.float)
    data.mc_theta = torch.tensor(track.get("mc_theta", float("nan")), dtype=torch.float)
    data.mc_lam   = torch.tensor(track.get("mc_lam",   float("nan")), dtype=torch.float)

    return data

# Expose only these names if someone does `from graphcreation import *`
__all__ = [
    "layer_map",
    "compute_lambda",
    "compute_all_lambdas_6hit",
    "compute_turning_angle",
    "compute_all_turning_angles_6hit",
    "get_layer_one_hot",
    "compute_total_turning_angle",
    "chord_length",
    "path_length",
    "straightness_ratio",
    "average_step_length",
    "signed_area",
    "extract_constraints6",
    "extract_geom_features_from_df6",
    "compute_global_edge_features_stats6",
    "track_to_graph6",
]

