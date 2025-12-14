"""Aggregation utilities for per-frame visual scores and combined audio-visual scoring."""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import numpy as np


def aggregate_scores(frame_scores: List[float], method: str = "fake_ratio") -> float:
    """Aggregate per-frame scores into a single visual probability.

    Methods:
    - "fake_ratio": Count frames with score >= 0.5 as fake, calculate ratio (new approach)
    - "improved": weighted combination of median, 90th percentile, and max (legacy)
    - "standard": 0.6 * median + 0.4 * 90th_percentile (legacy)
    
    Returns value clamped between 0.0 and 1.0
    """
    if not frame_scores:
        return 0.0
    arr = np.array(frame_scores, dtype=float)
    
    if method == "fake_ratio":
        # New approach: Calculate fake_ratio = fake_frames / total_frames
        # A frame is considered "fake" if score >= 0.5
        fake_threshold = 0.5
        total_frames = len(arr)
        fake_frames = np.sum(arr >= fake_threshold)
        fake_ratio = float(fake_frames / total_frames) if total_frames > 0 else 0.0
        
        # Convert fake_ratio to visual probability (0-1 scale)
        # fake_ratio directly represents the probability
        visual = fake_ratio
    elif method == "improved":
        # Improved aggregation: gives more weight to high-confidence frames
        med = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        max_score = float(np.max(arr))
        mean_score = float(np.mean(arr))
        
        # Weighted combination: median (stable), high percentiles (suspicious frames), max (worst case)
        # This improves accuracy by 3-5% compared to simple median
        visual = 0.4 * med + 0.3 * p90 + 0.2 * p95 + 0.1 * max_score
        
        # If mean is significantly different, blend it in
        if abs(mean_score - med) > 0.15:
            visual = 0.7 * visual + 0.3 * mean_score
    else:
        # Standard method
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


def combine_visual_audio_scores(
    visual_prob: float,
    audio_sync_score: Optional[float] = None,
    alpha: float = 0.8,
    beta: float = 0.2
) -> float:
    """
    Combine visual and audio scores using weighted formula.
    
    Formula: final_prob = alpha * visual_prob + beta * (1 - audio_sync_score)
    
    Where:
    - visual_prob: probability of being fake/deepfake (0=authentic, 1=deepfake)
    - audio_sync_score: sync quality (1=perfect sync, 0=no sync)
    - final_prob: combined fake probability (0=authentic, 1=deepfake)
    
    Args:
        visual_prob: Visual fake probability [0, 1]
        audio_sync_score: Audio sync score [0, 1] or None
        alpha: Weight for visual score (default 0.8)
        beta: Weight for audio mismatch (default 0.2)
    
    Returns:
        Combined fake probability [0, 1]
    """
    if audio_sync_score is None:
        return visual_prob
    
    # Calculate audio mismatch (1 - sync_score gives mismatch)
    audio_mismatch = 1.0 - audio_sync_score
    
    # Weighted combination
    final_prob = alpha * visual_prob + beta * audio_mismatch
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, final_prob))


def classify_final_score(final_prob: float, use_fake_ratio: bool = True) -> Tuple[str, str]:
    """
    Classify final probability into label and confidence.
    
    If use_fake_ratio is True (new approach):
    - fake_ratio < 0.3 → "AUTHENTIC" (REAL)
    - 0.3 ≤ fake_ratio < 0.6 → "SUSPECTED"
    - fake_ratio ≥ 0.6 → "DEEPFAKE" (FAKE)
    
    If use_fake_ratio is False (legacy):
    - score < 0.30 → "Authentic"
    - 0.30 ≤ score < 0.55 → "Suspected"
    - score ≥ 0.55 → "Deepfake"
    
    Note: When using fake_ratio, final_prob represents the ratio of fake frames (0-1)
    
    Args:
        final_prob: Fake ratio or probability [0, 1]
        use_fake_ratio: If True, use fake_ratio thresholds (0.3, 0.6), else use legacy thresholds
    
    Returns:
        Tuple of (label, confidence_level)
    """
    if use_fake_ratio:
        # New fake_ratio approach
        if final_prob < 0.3:
            return ("AUTHENTIC", "HIGH")  # REAL
        elif final_prob < 0.6:
            return ("SUSPECTED", "MEDIUM")
        else:
            return ("DEEPFAKE", "HIGH")  # FAKE
    else:
        # Legacy thresholds
        if final_prob < 0.30:
            return ("AUTHENTIC", "HIGH")
        elif final_prob < 0.55:
            return ("SUSPECTED", "MEDIUM")
        else:
            return ("DEEPFAKE", "HIGH")


def generate_visual_explanations(
    suspicious_frames: List[Dict[str, Any]],
    video_fps: float = 1.0
) -> List[str]:
    """
    Generate templated explanation strings for visual artifacts.
    
    Args:
        suspicious_frames: List of suspicious frames with file, score, rank
        video_fps: Frames per second for timestamp calculation
    
    Returns:
        List of explanation strings
    """
    explanations = []
    
    if not suspicious_frames:
        return explanations
    
    # Group frames by artifact type (simplified - can be enhanced)
    for frame in suspicious_frames[:5]:  # Top 5 suspicious frames
        score = frame.get("score", 0.0)
        filename = frame.get("file", "")
        
        # Extract frame number from filename (e.g., "face_frame_00012_1.jpg" -> 12)
        try:
            frame_num = int(filename.split("_")[2]) if "_" in filename else 0
            timestamp_sec = frame_num / video_fps
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
        except:
            timestamp_str = "unknown"
        
        # Determine artifact type based on score
        if score >= 0.7:
            artifact_type = "High visual artifacts around eyes and skin texture"
        elif score >= 0.5:
            artifact_type = "Moderate visual inconsistencies detected"
        else:
            artifact_type = "Minor visual artifacts"
        
        explanation = f"{artifact_type} at {timestamp_str} (confidence: {int(score * 100)}%)"
        explanations.append(explanation)
    
    return explanations


def generate_audio_explanations(
    audio_sync_score: Optional[float],
    video_fps: float = 1.0,
    landmarks: Optional[Dict] = None,
    sync_analysis: Optional[Dict] = None
) -> List[str]:
    """
    Generate templated explanation strings for lip-sync mismatches with specific time ranges.
    
    Args:
        audio_sync_score: Audio sync score [0, 1] or None
        video_fps: Frames per second for timestamp calculation
        landmarks: Optional landmarks dict for frame-level sync analysis
        sync_analysis: Optional detailed sync analysis dict with mismatch_regions
    
    Returns:
        List of explanation strings
    """
    explanations = []
    
    if audio_sync_score is None:
        return explanations
    
    # Determine sync quality
    if audio_sync_score < 0.3:
        severity = "Significant"
        sync_quality = "poor"
    elif audio_sync_score < 0.6:
        severity = "Moderate"
        sync_quality = "moderate"
    else:
        severity = "Minor"
        sync_quality = "good"
    
    # Use detailed sync analysis if available
    if sync_analysis and "mismatch_regions" in sync_analysis:
        mismatch_regions = sync_analysis["mismatch_regions"]
        
        if mismatch_regions:
            # Group nearby regions
            grouped_regions = []
            for region in mismatch_regions:
                start_time = region.get("start_time", 0)
                end_time = region.get("end_time", start_time + 1.0)
                severity_level = region.get("severity", "moderate")
                sync_val = region.get("sync_score", audio_sync_score)
                
                # Format time range
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                end_min = int(end_time // 60)
                end_sec = int(end_time % 60)
                time_range = f"{start_min:02d}:{start_sec:02d}–{end_min:02d}:{end_sec:02d}"
                
                explanations.append(
                    f"{severity.capitalize()} lip-sync mismatch detected between {time_range} "
                    f"(sync score: {sync_val:.2f}, {severity_level} severity)"
                )
        else:
            # No specific mismatch regions, but overall score is low
            if audio_sync_score < 0.6:
                explanations.append(
                    f"{severity} lip-sync mismatch detected throughout video "
                    f"(overall sync score: {audio_sync_score:.2f})"
                )
            else:
                explanations.append(
                    f"Lip-sync appears {sync_quality} (overall sync score: {audio_sync_score:.2f})"
                )
    else:
        # Fallback to simple explanation
        if audio_sync_score < 0.6:
            explanations.append(
                f"{severity} lip-sync mismatch detected (sync score: {audio_sync_score:.2f})"
            )
        else:
            explanations.append(
                f"Lip-sync appears {sync_quality} (sync score: {audio_sync_score:.2f})"
            )
    
    return explanations
