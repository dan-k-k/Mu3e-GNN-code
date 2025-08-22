# build6hittracks.py
# # 6hit graph generation (I call it validation as they are not yet saved as graphs with features)
# # what this does exactly:
# # 1.  hits are paired by layer: hits in layer 1 and 2 together
# #                             hits in layer 3 and 4 together
# #                             hits in layer 4+ and 3+ or 4- and 3- together.
# 
# # 2.  to begin, pairs of hits are found by enforcing a simple transverse distance constraint.
# #           this greatly reduces the number of possible combinations early, important as
# #           later constraints are more expensive and slow. 
# #           ideally, expensive constraints are not used at all, but this can result
# #           in too many validated fakes or reduced graph generation efficiency. 
# #           the 8hit track 'validation' method applies many inexpensive constraints hit-by-hit to reduce as many
# #           candidates as possible early. 8hits are tougher as the number of combinations (candidates) explodes
# #           in cases where a particle has very low z momentum and many repeated recurls.
# #           the 8hit method is in another notebook.
# #           **6hit graph generation efficiency for the method below is around 99% for all four datasets.** loss of efficiency occurs with 
# #           bending centre and z ratio tolerances.
# 
# # 3.  then, the pairs are combined (cartesian product) into all available 6hit candidates. ignore
# #     candidates that use the same hit twice (useful for 8hit generation when returning to the same layer). 
# 
# # 4.  check z-difference ratios between pairs. rather than a hard limit on the allowed z distance between two hits, 
# #     compare z-difference ratios of the pairs. the constraint for comparing inner and outer layers does not follow 
# #     the physics properly and simply uses an arbitrary factor below of 0.4. however, it is loose enough to accept almost
# #     all true candidates. (eg. particles with lower transverse momentum will have a smaller factor than 0.4).
# #     note: when building 8hit graphs, it is important to also include an absolute buffer to mitigate MS impact on low z momentum particles. 
# #   
# # 5.  then, filter by the more expensive constraints, which are:
# #           bending centres:
# #               for pairs of hits, find the intersection of bisectors.
# #               as there are three pairs, three centres can be estimated.
# #               'validation' happens when two of the three centres match to a *fixed* tolerance
# #               (a refined method would find a way to scale with transverse momentum).
# #           pitches:
# #               for pairs of hits, divide the changes in z of pair midpoints by the change in angle phi (of the lines that connect two hits).
# #               the midpoint of the two hits is tangent to the angle phi of the bending circle so it is a reasonable approximation.
# #               the angle between the first two pairs is assumed to never be larger than 90deg.
# #               therefore, the rotation direction is found from the wrapped (-90 to 90deg) angle from the first two pairs
# #               and the later pairs including layers '5' and '6' follow the rotation direction for angles -360 to 360deg.
# #               pitch is signed.
# #               'validation' happens when two of the three pitches match to a *fixed* tolerance
# #               (a refined method would find a way to scale with z momentum).

import numpy as np
from itertools import product
from sklearn.cluster import DBSCAN
import pandas as pd
import logging
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimate_bending_center(pair1, pair2):
    def get_perpendicular_bisector(hit_a, hit_b):
        x1, y1 = hit_a['x'], hit_a['y']
        x2, y2 = hit_b['x'], hit_b['y']

        # Midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Slope of the original line
        delta_x = x2 - x1
        delta_y = y2 - y1
        if delta_x == 0:
            # Original line is vertical, so bisector is horizontal
            return (0, mid_y)
        if delta_y == 0:
            # Original line is horizontal, so bisector is vertical
            return (np.inf, mid_x)

        slope_original = delta_y / delta_x
        slope_bisector = -1 / slope_original
        intercept_bisector = mid_y - slope_bisector * mid_x

        return (slope_bisector, intercept_bisector)

    # Compute perpendicular bisectors for both pairs
    bisector1 = get_perpendicular_bisector(pair1[0], pair1[1])
    bisector2 = get_perpendicular_bisector(pair2[0], pair2[1])

    # Check for vertical bisectors
    if bisector1[0] == np.inf and bisector2[0] == np.inf:
        return (np.nan, np.nan)  # Parallel vertical bisectors, no intersection
    elif bisector1[0] == np.inf:
        x_center = bisector1[1]
        y_center = bisector2[0] * x_center + bisector2[1]
    elif bisector2[0] == np.inf:
        x_center = bisector2[1]
        y_center = bisector1[0] * x_center + bisector1[1]
    elif bisector1[0] == bisector2[0]:
        return (np.nan, np.nan)  # Parallel bisectors, no unique intersection
    else:
        # Solve for intersection
        slope1, intercept1 = bisector1
        slope2, intercept2 = bisector2
        x_center = (intercept2 - intercept1) / (slope1 - slope2)
        y_center = slope1 * x_center + intercept1

    return (x_center, y_center)

def estimate_three_bending_centers(track_hits):
    if len(track_hits) != 6:
        logger.warning("Invalid number of hits provided for bending center estimation.")
        return [(np.nan, np.nan)] * 3, [np.nan] * 3

    # Define hit pairs for center estimation
    # Center 1: Layer1 & Layer2 with Layer3 & Layer4
    pair1_a = (track_hits[0], track_hits[1])  # Layer1 & Layer2
    pair1_b = (track_hits[2], track_hits[3])  # Layer3 & Layer4

    # Center 2: Layer1 & Layer2 with Layer5 & Layer6
    pair2_a = (track_hits[0], track_hits[1])  # Layer1 & Layer2
    pair2_b = (track_hits[4], track_hits[5])  # Layer5 & Layer6

    # Center 3: Layer3 & Layer4 with Layer5 & Layer6
    pair3_a = (track_hits[2], track_hits[3])  # Layer3 & Layer4
    pair3_b = (track_hits[4], track_hits[5])  # Layer5 & Layer6

    # Estimate centers
    center1 = estimate_bending_center(pair1_a, pair1_b)
    center2 = estimate_bending_center(pair2_a, pair2_b)
    center3 = estimate_bending_center(pair3_a, pair3_b)

    centers = [center1, center2, center3]

    # Calculate radii based on centers
    def calculate_radius(center, hit):
        if np.isnan(center[0]) or np.isnan(center[1]):
            return np.nan
        x_c, y_c = center
        x, y = hit['x'], hit['y']
        return np.sqrt((x - x_c)**2 + (y - y_c)**2)

    radii = []
    # Radius1 based on Layer1 hit
    R1 = calculate_radius(center1, track_hits[0])
    radii.append(R1)

    # Radius2 based on Layer1 hit (consistent with pair2_a)
    R2 = calculate_radius(center2, track_hits[0])
    radii.append(R2)

    # Radius3 based on Layer3 hit
    R3 = calculate_radius(center3, track_hits[2])
    radii.append(R3)

    return centers, radii

def estimate_pitch(pair_a, pair_b, rotation_direction=None, limit_angle_diff=False):
    def slope_angle(hit_a, hit_b):
        # Compute the angle between two points
        delta_x = hit_b['x'] - hit_a['x']
        delta_y = hit_b['y'] - hit_a['y']
        return np.arctan2(delta_y, delta_x)  # Angle in radians

    # Calculate angles of the pairs
    angle_a = slope_angle(pair_a[0], pair_a[1])
    angle_b = slope_angle(pair_b[0], pair_b[1])

    # Calculate raw angle difference
    angle_diff = angle_b - angle_a

    # Apply constraint for the first angle difference if specified
    if limit_angle_diff:
        # Wrap angle_diff to [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Constrain to [-π/2, π/2] for the smallest angle difference
        if angle_diff > np.pi / 2:
            angle_diff -= np.pi
        elif angle_diff < -np.pi / 2:
            angle_diff += np.pi

    # Account for rotation direction only for pitch 2 and 3
    if rotation_direction is not None:
        if rotation_direction > 0:  # anti-clockwise !!!
            if angle_diff < 0:
                angle_diff += 2 * np.pi
        else:  # Clockwise
            if angle_diff > 0:
                angle_diff -= 2 * np.pi

    # Avoid division by zero for pitch calculation
    if angle_diff == 0:
        return np.nan, angle_diff, 0

    # Calculate midpoints for z-difference
    def midpoint(hit1, hit2):
        return (
            (hit1['x'] + hit2['x']) / 2,
            (hit1['y'] + hit2['y']) / 2,
            (hit1.get('z', 0) + hit2.get('z', 0)) / 2  # Ensure 'z' exists
        )

    mid_a = midpoint(pair_a[0], pair_a[1])
    mid_b = midpoint(pair_b[0], pair_b[1])

    z_diff = mid_b[2] - mid_a[2]

    # Calculate pitch as z_diff per radian
    pitch = z_diff / angle_diff  # Units of z per radian

    return pitch, angle_diff, z_diff

def estimate_three_pitches(track_hits):
    if len(track_hits) != 6:
        logger.warning("Invalid number of hits provided for pitch estimation.")
        return [np.nan] * 3, [np.nan] * 3, [np.nan] * 3
    # Define hit pairs for pitch estimation
    pair1_a = (track_hits[0], track_hits[1])  # Layer1 & Layer2
    pair1_b = (track_hits[2], track_hits[3])  # Layer3 & Layer4
    pair2_a = (track_hits[0], track_hits[1])  # Layer1 & Layer2
    pair2_b = (track_hits[4], track_hits[5])  # Layer5 & Layer6
    pair3_a = (track_hits[2], track_hits[3])  # Layer3 & Layer4
    pair3_b = (track_hits[4], track_hits[5])  # Layer5 & Layer6

    # Estimate Pitch1 and its angle difference
    pitch1, angle_diff1, z_diff1 = estimate_pitch(pair1_a, pair1_b, limit_angle_diff=True)

    # Determine rotation direction based on Pitch1's angle difference
    rotation_direction = 1 if angle_diff1 > 0 else -1

    # Estimate Pitch2 using the determined rotation direction
    pitch2, angle_diff2, z_diff2 = estimate_pitch(pair2_a, pair2_b, rotation_direction=rotation_direction, limit_angle_diff=False)

    # Estimate Pitch3 using the determined rotation direction
    pitch3, angle_diff3, z_diff3 = estimate_pitch(pair3_a, pair3_b, rotation_direction=rotation_direction, limit_angle_diff=False)

    pitches = [pitch1, pitch2, pitch3]
    angle_diffs = [angle_diff1, angle_diff2, angle_diff3]
    z_diffs = [z_diff1, z_diff2, z_diff3]

    return pitches, angle_diffs, z_diffs

def build_and_validate_tracks(
    hits_df_unique, frame_id, layer_sequences, 
    center_tolerance=50, radius_tolerance=50, 
    pitch_tolerance=40,
    distance_constraints={
        '1-2': 30,
        '3-4': 60,
        '5-6': 60
    },
    z_distance_error_margin=0.65  # 20% error margin
):
    validated_tracks = []
    seen_candidates = set()  # To deduplicate candidates by sorted hit IDs
    # Filter hits for the current frame
    frame_hits = hits_df_unique[hits_df_unique['frameId'] == frame_id]
    
    # Loop over each provided layer sequence
    for layer_sequence in layer_sequences:
        # Unpack the six layers from the sequence
        l1, l2, l3, l4, l5, l6 = layer_sequence
        # Get hits per layer
        hits_1 = frame_hits[frame_hits['layer'] == l1].to_dict('records')
        hits_2 = frame_hits[frame_hits['layer'] == l2].to_dict('records')
        hits_3 = frame_hits[frame_hits['layer'] == l3].to_dict('records')
        hits_4 = frame_hits[frame_hits['layer'] == l4].to_dict('records')
        hits_5 = frame_hits[frame_hits['layer'] == l5].to_dict('records')
        hits_6 = frame_hits[frame_hits['layer'] == l6].to_dict('records')
        
        # Precompute valid pairs for each group based on simple distance constraints.
        valid_pairs_12 = []
        for hit1 in hits_1:
            for hit2 in hits_2:
                d12 = np.hypot(hit1['x'] - hit2['x'], hit1['y'] - hit2['y'])
                if d12 <= distance_constraints['1-2']:
                    valid_pairs_12.append((hit1, hit2))
        
        valid_pairs_34 = []
        for hit3 in hits_3:
            for hit4 in hits_4:
                d34 = np.hypot(hit3['x'] - hit4['x'], hit3['y'] - hit4['y'])
                if d34 <= distance_constraints['3-4']:
                    valid_pairs_34.append((hit3, hit4))
        
        valid_pairs_56 = []
        for hit5 in hits_5:
            for hit6 in hits_6:
                d56 = np.hypot(hit5['x'] - hit6['x'], hit5['y'] - hit6['y'])
                if d56 <= distance_constraints['5-6']:
                    valid_pairs_56.append((hit5, hit6))
        
        # Iterate over the product of the valid pairs to form full 6–hit candidates.
        from itertools import product
        for pair12, pair34, pair56 in product(valid_pairs_12, valid_pairs_34, valid_pairs_56):
            # Combine the pairs into a full candidate track.
            candidate = pair12 + pair34 + pair56
            
            # Ensure uniqueness of hits (no duplicates)
            hit_ids = [hit['hit_id'] for hit in candidate]
            if len(set(hit_ids)) != len(hit_ids):
                continue  # Skip candidate with duplicate hits
            
            # Deduplicate candidate: sort hit IDs and use as key.
            candidate_key = tuple(sorted(hit_ids))
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            
            # Re-check inexpensive distance constraints.
            distance_12 = np.hypot(pair12[0]['x'] - pair12[1]['x'], pair12[0]['y'] - pair12[1]['y'])
            distance_34 = np.hypot(pair34[0]['x'] - pair34[1]['x'], pair34[0]['y'] - pair34[1]['y'])
            distance_56 = np.hypot(pair56[0]['x'] - pair56[1]['x'], pair56[0]['y'] - pair56[1]['y'])
            
            # Check z-differences for each pair.
            z_diff_12 = pair12[0]['z'] - pair12[1]['z']
            z_diff_34 = pair34[0]['z'] - pair34[1]['z']
            z_diff_56 = pair56[0]['z'] - pair56[1]['z']
            if z_diff_34 != 0:
                lower_bound_z56 = (1 - z_distance_error_margin) * z_diff_34
                upper_bound_z56 = (1 + z_distance_error_margin) * z_diff_34
                lower_bound_z56, upper_bound_z56 = sorted([lower_bound_z56, upper_bound_z56])
                if not (lower_bound_z56 <= z_diff_56 <= upper_bound_z56):
                    continue
            else:
                if z_diff_56 != 0:
                    continue
            
            # --- New: Apply constraint on z_diff_12 relative to z_diff_34 ---
            if z_diff_34 == 0:
                if z_diff_12 != 0:
                    continue
            else:
                desired_z_diff_12 = 0.4 * z_diff_34
                lower_bound_z12 = (1 - z_distance_error_margin/0.8) * desired_z_diff_12
                upper_bound_z12 = (1 + z_distance_error_margin/0.8) * desired_z_diff_12
                lower_bound_z12, upper_bound_z12 = sorted([lower_bound_z12, upper_bound_z12])
                if not (lower_bound_z12 <= z_diff_12 <= upper_bound_z12):
                    continue
            # -------------------------------------------------------------
            
            # Now perform the more expensive calculations.
            centers, radii = estimate_three_bending_centers(candidate)
            pitches, angle_diffs, z_diffs = estimate_three_pitches(candidate)
            
            # Validate that all estimated centers and radii are finite.
            centers_finite = [np.isfinite(c[0]) and np.isfinite(c[1]) for c in centers]
            radii_finite = [np.isfinite(r) for r in radii]
            if not (all(centers_finite) and all(radii_finite)):
                continue
            
            # Collect valid centers and radii.
            valid_centers = []
            valid_radii = []
            for c, r in zip(centers, radii):
                if np.isfinite(c[0]) and np.isfinite(c[1]) and np.isfinite(r):
                    valid_centers.append(c)
                    valid_radii.append(r)
            
            if len(valid_centers) < 2:
                continue  # Need at least two valid center estimations.
            
            # Collect valid pitches.
            valid_pitches = [p for p in pitches if np.isfinite(p)]
            if len(valid_pitches) < 2:
                continue  # Need at least two valid pitch values.
            
            # Compare all pairs of centers to find matching pairs.
            center_matched_pairs = []
            for i in range(len(valid_centers)):
                for j in range(i + 1, len(valid_centers)):
                    center_distance = np.sqrt((valid_centers[i][0] - valid_centers[j][0])**2 +
                                              (valid_centers[i][1] - valid_centers[j][1])**2)
                    radius_diff = abs(valid_radii[i] - valid_radii[j])
                    if center_distance <= center_tolerance and radius_diff <= radius_tolerance:
                        center_matched_pairs.append(((valid_centers[i], valid_centers[j]),
                                                     (valid_radii[i], valid_radii[j])))
            if not center_matched_pairs:
                continue
            
            # Compare all pairs of valid pitches to find matching pairs.
            pitch_matched = False
            for i in range(len(valid_pitches)):
                for j in range(i + 1, len(valid_pitches)):
                    if abs(valid_pitches[i] - valid_pitches[j]) <= pitch_tolerance:
                        pitch_matched = True
                        break
                if pitch_matched:
                    break
            if not pitch_matched:
                continue
            
            # Select the first matching center pair and pitch pair.
            selected_centers, selected_radii = center_matched_pairs[0][0], center_matched_pairs[0][1]
            selected_pitches = []
            for i in range(len(valid_pitches)):
                for j in range(i + 1, len(valid_pitches)):
                    if abs(valid_pitches[i] - valid_pitches[j]) <= pitch_tolerance:
                        selected_pitches = [valid_pitches[i], valid_pitches[j]]
                        break
                if selected_pitches:
                    break
            if not selected_pitches:
                continue
            
            # If all validations pass, add the candidate track.
            validated_tracks.append({
                'frameId': frame_id,
                'hits': candidate,
                'centers': centers,      # List of three centers
                'radii': radii,          # List of three radii
                'pitches': pitches,      # List of three pitches
                'angle_diffs': angle_diffs,
                'z_diffs': z_diffs,
                'transverse_distances': {
                    'd_12': distance_12,
                    'd_34': distance_34,
                    'd_56': distance_56
                },
                'z_distances': {
                    'z_diff_12': z_diff_12,
                    'z_diff_34': z_diff_34,
                    'z_diff_56': z_diff_56
                }
            })
    
    return validated_tracks

# the below code is for plotting and finding constraints 
# that can be used to find all of the available true 6hit tracks.

def compute_center_radius_min(track_hits):
    """
    Given `track_hits` as a list of 6 dictionaries,
    each dict has keys 'x','y','z'.  
    Returns (min_center_distance, min_radius_difference).
    """
    centers, radii = estimate_three_bending_centers(track_hits)
    valid = [
        (c, r) for c, r in zip(centers, radii)
        if np.isfinite(c[0]) and np.isfinite(c[1]) and np.isfinite(r)
    ]
    if len(valid) < 2:
        return np.nan, np.nan

    cdiffs = []
    rdiffs = []
    for (c1, r1), (c2, r2) in combinations(valid, 2):
        cdiffs.append(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
        rdiffs.append(abs(r1 - r2))

    return min(cdiffs), min(rdiffs)

def compute_pitch_min(track_hits):
    """
    Given `track_hits` as a list of 6 dictionaries,
    each dict has keys 'x','y','z'.  
    Returns the minimum absolute pitch difference.
    """
    pitches, _, _ = estimate_three_pitches(track_hits)
    valid_p = [p for p in pitches if np.isfinite(p)]
    if len(valid_p) < 2:
        return np.nan
    diffs = []
    for p1, p2 in combinations(valid_p, 2):
        diffs.append(abs(p1 - p2))
    return min(diffs)

def collect_sixhit_distributions(sixhit_tracks_df):
    """
    Given a DataFrame `sixhit_tracks_df` whose 'hits' column is a
    length-6 Python list of (x,y,z) tuples, this function will loop
    over every row and return a dict of lists:
      {
        'd12': [ ... ],
        'd34': [ ... ],
        'd56': [ ... ],
        'zratio': [ ... ],      # (|z5−z6| / |z3−z4| * 100)
        'center_diff': [ ... ],  # min center‐center distance
        'radius_diff': [ ... ],  # min radius difference
        'pitch_diff': [ ... ]    # min pitch difference
      }
    """
    d12_list      = []
    d34_list      = []
    d56_list      = []
    zratio56_list   = []
    zratio12_list = []
    center_list   = []
    radius_list   = []
    pitch_list    = []

    for _, row in sixhit_tracks_df.iterrows():
        # Step A: Convert each tuple (x,y,z) to a dict {'x':…, 'y':…, 'z':…}
        hits_as_dicts = [
            {'x': x, 'y': y, 'z': z}
            for (x, y, z) in row['hits']
        ]

        # transverse distances
        d12 = np.hypot(
            hits_as_dicts[0]['x'] - hits_as_dicts[1]['x'],
            hits_as_dicts[0]['y'] - hits_as_dicts[1]['y']
        )
        d34 = np.hypot(
            hits_as_dicts[2]['x'] - hits_as_dicts[3]['x'],
            hits_as_dicts[2]['y'] - hits_as_dicts[3]['y']
        )
        d56 = np.hypot(
            hits_as_dicts[4]['x'] - hits_as_dicts[5]['x'],
            hits_as_dicts[4]['y'] - hits_as_dicts[5]['y']
        )

        # z‐ratio = |z5−z6| / |z3−z4| × 100%
        dz34 = abs(hits_as_dicts[2]['z'] - hits_as_dicts[3]['z'])
        dz56 = abs(hits_as_dicts[4]['z'] - hits_as_dicts[5]['z'])
        dz12 = abs(hits_as_dicts[0]['z'] - hits_as_dicts[1]['z'])
        zr56  = (dz56 / dz34 * 100.0) if dz34 > 0 else np.nan
        zr12 = (dz12 / dz34 * 100.0) if dz34 > 0 else np.nan

        # circle centers & radii
        c_diff, r_diff = compute_center_radius_min(hits_as_dicts)

        # pitch
        p_diff = compute_pitch_min(hits_as_dicts)

        # append to lists
        d12_list.append(d12)
        d34_list.append(d34)
        d56_list.append(d56)
        zratio56_list.append(zr56)
        zratio12_list.append(zr12)
        center_list.append(c_diff)
        radius_list.append(r_diff)
        pitch_list.append(p_diff)

    return {
        "d12": d12_list,
        "d34": d34_list,
        "d56": d56_list,
        "zratio56": zratio56_list,
        "zratio12": zratio12_list,
        "center_diff": center_list,
        "radius_diff": radius_list,
        "pitch_diff": pitch_list
    }

def unpack_xyz(hit):
    """
    Given a single hit, which can be either:
      - a tuple (x, y, z), or
      - a dict with keys 'x','y','z',
    return the triple (x, y, z) as floats.
    """
    if isinstance(hit, dict):
        return hit["x"], hit["y"], hit["z"]
    else:
        # assume a tuple or list of length ≥ 3
        return hit[0], hit[1], hit[2]
    
def row_constraints(row):
    """
    Given one row of `sixhit_tracks_df`, where row['hits'] is a list of six
    entries (each entry either a dict {'x':…, 'y':…, 'z':…} or a tuple (x,y,z)),
    compute all nine constraint‐values and return as a dict:
       - d12, d34, d56  (transverse distances)
       - z12, z34, z56  (z-differences)
       - center_diff, radius_diff  (min‐center, min‐radius)
       - pitch_diff              (min‐pitch difference)
    """
    track_hits = row["hits"]  # length‐6 list, each either dict or tuple

    # 1) Unpack (x0, y0, z0), (x1, y1, z1), … using unpack_xyz(...)
    x0, y0, z0 = unpack_xyz(track_hits[0])
    x1, y1, z1 = unpack_xyz(track_hits[1])
    x2, y2, z2 = unpack_xyz(track_hits[2])
    x3, y3, z3 = unpack_xyz(track_hits[3])
    x4, y4, z4 = unpack_xyz(track_hits[4])
    x5, y5, z5 = unpack_xyz(track_hits[5])

    # 2) Compute transverse distances:
    d12 = np.hypot(x0 - x1, y0 - y1)
    d34 = np.hypot(x2 - x3, y2 - y3)
    d56 = np.hypot(x4 - x5, y4 - y5)

    # 3) Compute z‐differences:
    z12 = z0 - z1
    z34 = z2 - z3
    z56 = z4 - z5

    # 4) Build a list of dicts (so compute_center_radius_min & compute_pitch_min still work):
    hits_as_dicts = []
    for h in track_hits:
        xx, yy, zz = unpack_xyz(h)
        hits_as_dicts.append({"x": xx, "y": yy, "z": zz})

    center_diff, radius_diff = compute_center_radius_min(hits_as_dicts)
    pitch_diff           = compute_pitch_min(hits_as_dicts)

    return {
        "d12":         d12,
        "d34":         d34,
        "d56":         d56,
        "z12":         z12,
        "z34":         z34,
        "z56":         z56,
        "center_diff": center_diff,
        "radius_diff": radius_diff,
        "pitch_diff":  pitch_diff
    }

