# determinerealorfake.py

"""
Utility functions for deciding whether a 6-hit “validated track” is real or fake,
and for extracting consistent MC parameters (type, pT, angles).

All functions operate on a single track (a list of six hit-dicts) and a
corresponding hit→MC map (passed in as an argument).  
"""

from typing import List, Dict, Any, Optional, Union

# Helper functions to aggregate per‐track
#       all_hits_zero_tid is needed to completely discard cases where all 6 hits have mc_tid=0,
#       as this can sometimes be a truth track from sixhit_tracks_df but with no information.
#       and we do not want these to be saved as fakes. any future validated track with at least one
#       mc_tid=0 is fake, then this simplifies the method for finding matching mc_tids.

def all_hits_zero_tid(
    track_hits: List[Dict[str, Any]],
    hit_to_tid_map: Dict[str, int]
) -> bool:
    """
    Return True if *every* hit in this 6-hit track has mc_tid == 0.
    (Used to discard pure-noise candidates before calling anything else.)
    """
    return all(hit_to_tid_map.get(h["hit_id"], 0) == 0 for h in track_hits)


def determine_track_tid(
    track_hits: List[Dict[str, Any]],
    hit_to_tid_map: Dict[str, int]
) -> Union[int, List[int]]:
    """
    Given a 6-element list of hit-dicts, return:
      - a single non-zero mc_tid (int) if all six share the same non-zero TID,
      - 0                        if *any* hit has mc_tid == 0, or if there is disagreement,
      - a Python list of multiple non-zero TIDs if more than one distinct non-zero appears.
    
    track_hits:      [ { "hit_id": str, … }, … ]  (exactly 6 hits)
    hit_to_tid_map:  { hit_id_str → mc_tid (int) }
    """
    mc_tids = [hit_to_tid_map.get(h["hit_id"], 0) for h in track_hits]

    # 1) If any hit has mc_tid == 0, mark the entire track as fake (return 0)
    if any(tid == 0 for tid in mc_tids):
        return 0

    # 2) All six are nonzero → check if they all agree
    unique_tids = set(mc_tids)
    if len(unique_tids) == 1:
        return unique_tids.pop()       # perfectly real track
    else:
        # disagreement among nonzero IDs → treat as fake, but enumerate the mismatched IDs
        return list(unique_tids)


def determine_track_type(
    track_hits: List[Dict[str, Any]],
    hit_to_type_map: Dict[str, Union[str, int]]
) -> str:
    """
    Given a 6-element list of hit-dicts, look up each hit’s mc_type (string or int).
    Return:
      - the single non-Unknown mc_type if they all agree and none are 'Unknown',
      - 'Multiple' if more than one distinct non-Unknown mc_type appears,
      - 'Unknown' if every hit is mapped to 'Unknown' or the map is missing.
    
    track_hits:       [ { "hit_id": str, … }, … ]
    hit_to_type_map:  { hit_id_str → mc_type (int or str) }
    """
    mc_types = [hit_to_type_map.get(h["hit_id"], "Unknown") for h in track_hits]
    # If any hit’s type is explicitly 'Unknown', we cannot trust it → return 'Unknown'
    if any(t == "Unknown" for t in mc_types):
        return "Unknown"

    unique_types = set(mc_types)
    if len(unique_types) == 1:
        return unique_types.pop()
    else:
        return "Multiple"


def determine_track_pt(
    track_hits: List[Dict[str, Any]],
    tol: float = 1e-5
) -> Optional[tuple]:
    """
    Check that every hit in the track has identical (mc_pt, mc_p). If so, return (mc_pt, mc_p).
    Otherwise print a warning and return None.

    track_hits:  [ { "hit_id": str, "mc_pt": float, "mc_p": float, … }, … ]
    tol:         maximum allowed difference among pt (and p) values.
    """
    pts = [h.get("mc_pt") for h in track_hits if h.get("mc_pt") is not None]
    ps  = [h.get("mc_p")  for h in track_hits if h.get("mc_p")  is not None]

    if not pts or not ps:
        return None

    first_pt, first_p = pts[0], ps[0]

    if any(abs(x - first_pt) > tol for x in pts[1:]):
        print("Warning: inconsistent mc_pt in track", pts)
        return None

    if any(abs(x - first_p) > tol for x in ps[1:]):
        print("Warning: inconsistent mc_p in track", ps)
        return None

    return first_pt, first_p


def angle_from_map(
    track_hits: List[Dict[str, Any]],
    angle_name: str,
    hit_angle_map: Dict[str, Dict[str, float]],
    tol: float = 1e-6
) -> Optional[float]:
    """
    Given a 6-hit track, look up each hit’s 'mc_phi' or 'mc_theta' or 'mc_lam' via hit_angle_map.
    Return the single consistent value if all are the same (within tol), otherwise print a warning and return None.

    track_hits:    [ { "hit_id": str, … }, … ]
    angle_name:    one of "mc_phi", "mc_theta", "mc_lam"
    hit_angle_map: { hit_id_str → { "mc_phi":…, "mc_theta":…, "mc_lam":… } }
    tol:           allowed numeric difference among angle values
    """
    vals = []
    for h in track_hits:
        hid = h["hit_id"]
        if hid in hit_angle_map:
            vals.append(hit_angle_map[hid][angle_name])

    if not vals:
        return None

    first = vals[0]
    if any(abs(v - first) > tol for v in vals[1:]):
        print(f"Warning: {angle_name} varies in track", vals)
        return None

    return first


__all__ = [
    "all_hits_zero_tid",
    "determine_track_tid",
    "determine_track_type",
    "determine_track_pt",
    "angle_from_map",
]

