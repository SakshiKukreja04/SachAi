"""Aggregation utilities for per-frame visual scores."""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np


def aggregate_scores(frame_scores: List[float]) -> float:
    """Aggregate per-frame scores into a single visual probability.

    Formula: visual = 0.6 * median + 0.4 * 90th_percentile
    Returns value clamped between 0.0 and 1.0
    """
    if not frame_scores:
        return 0.0
    arr = np.array(frame_scores, dtype=float)
    med = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    visual = 0.6 * med + 0.4 * p90
    visual = max(0.0, min(1.0, visual))
    return visual


def top_k_frames(frame_scores: List[float], filenames: List[str], k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k suspicious frames as list of dicts with file, score and rank (1-based).

    The returned list is ordered by rank (1 = most suspicious).
    """
    if not frame_scores:
        return []
    paired = list(zip(filenames, frame_scores))
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    topk = paired_sorted[:k]
    result = []
    for i, (fname, score) in enumerate(topk, start=1):
        result.append({"file": fname, "score": float(score), "rank": i})
    return result
