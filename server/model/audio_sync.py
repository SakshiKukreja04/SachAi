"""
Audio-visual lip-sync mismatch detector using heuristic approach.

Extracts audio features (MFCC/envelope) and mouth opening measures from landmarks,
then computes correlation to derive a sync score.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not available. Install with: pip install librosa")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: opencv-python not available.")


def extract_audio_features(audio_path: str, window_size: float = 0.3, hop_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract audio envelope and MFCC features for time windows.
    
    Args:
        audio_path: Path to audio file (WAV)
        window_size: Window size in seconds (default 0.3s)
        hop_size: Hop size in seconds (default 0.1s)
    
    Returns:
        Tuple of (time_points, audio_energy) where:
        - time_points: Array of time points in seconds
        - audio_energy: Array of audio energy values per window
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio feature extraction. Install with: pip install librosa")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz for consistency
    
    # Compute short-time energy (envelope)
    frame_length = int(window_size * sr)
    hop_length = int(hop_size * sr)
    
    # Compute RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Compute time points
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Normalize energy to [0, 1]
    if rms.max() > 0:
        rms = rms / rms.max()
    
    return times, rms


def compute_mouth_opening(landmarks: Dict, frame_idx: int, face_idx: int = 0) -> Optional[float]:
    """
    Compute mouth opening measure from facial landmarks.
    
    Args:
        landmarks: Landmarks dictionary with structure:
            {
                "frames": [
                    {
                        "frame": "frame_00001.jpg",
                        "faces": [
                            {
                                "mouth_landmarks": [[x1, y1], [x2, y2], ...]  # 20 points for mouth
                                # OR
                                "mouth_top": [x, y],
                                "mouth_bottom": [x, y],
                                "mouth_left": [x, y],
                                "mouth_right": [x, y]
                            }
                        ]
                    }
                ]
            }
        frame_idx: Index of frame (0-based)
        face_idx: Index of face in frame (default 0)
    
    Returns:
        Mouth opening measure (distance between top and bottom lip) or None if not available
    """
    try:
        if frame_idx >= len(landmarks.get("frames", [])):
            return None
        
        frame_data = landmarks["frames"][frame_idx]
        faces = frame_data.get("faces", [])
        
        if face_idx >= len(faces):
            return None
        
        face_data = faces[face_idx]
        
        # Check for mouth_landmarks (mediapipe format: 20 points)
        if "mouth_landmarks" in face_data:
            mouth_points = np.array(face_data["mouth_landmarks"])
            if len(mouth_points) < 4:
                return None
            
            # Get top and bottom lip points
            # Mediapipe mouth landmarks: points 13-14 are top lip center, 17-18 are bottom lip center
            # Or use min/max y coordinates
            top_lip_y = mouth_points[:, 1].min()  # Minimum y (top of mouth)
            bottom_lip_y = mouth_points[:, 1].max()  # Maximum y (bottom of mouth)
            mouth_opening = abs(bottom_lip_y - top_lip_y)
            
            # Normalize by face size if available
            if "bbox" in face_data:
                bbox = face_data["bbox"]
                face_height = bbox[3] if len(bbox) >= 4 else 1.0
                if face_height > 0:
                    mouth_opening = mouth_opening / face_height
            
            return float(mouth_opening)
        
        # Check for explicit mouth coordinates
        elif "mouth_top" in face_data and "mouth_bottom" in face_data:
            top = np.array(face_data["mouth_top"])
            bottom = np.array(face_data["mouth_bottom"])
            mouth_opening = np.linalg.norm(bottom - top)
            
            # Normalize by face size if available
            if "bbox" in face_data:
                bbox = face_data["bbox"]
                face_height = bbox[3] if len(bbox) >= 4 else 1.0
                if face_height > 0:
                    mouth_opening = mouth_opening / face_height
            
            return float(mouth_opening)
        
        # Fallback: estimate from bbox (less accurate)
        elif "bbox" in face_data:
            bbox = face_data["bbox"]
            # Estimate mouth opening as a fraction of face height (rough approximation)
            face_height = bbox[3] if len(bbox) >= 4 else 0
            if face_height > 0:
                # Typical mouth opening is 5-15% of face height
                return face_height * 0.1  # Use a constant estimate
            return None
        
        return None
    
    except Exception as e:
        print(f"Error computing mouth opening: {e}")
        return None


def extract_mouth_opening_timeline(landmarks: Dict, video_fps: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mouth opening measures over time from landmarks.
    
    Args:
        landmarks: Landmarks dictionary
        video_fps: Frames per second of the video
    
    Returns:
        Tuple of (time_points, mouth_opening_values)
    """
    frames = landmarks.get("frames", [])
    if not frames:
        return np.array([]), np.array([])
    
    times = []
    mouth_values = []
    
    for idx, frame_data in enumerate(frames):
        time_sec = idx / video_fps
        mouth_opening = compute_mouth_opening(landmarks, idx, face_idx=0)
        
        if mouth_opening is not None:
            times.append(time_sec)
            mouth_values.append(mouth_opening)
    
    return np.array(times), np.array(mouth_values)


def compute_correlation(audio_energy: np.ndarray, mouth_opening: np.ndarray, 
                        audio_times: np.ndarray, mouth_times: np.ndarray) -> float:
    """
    Compute correlation between audio energy and mouth opening.
    
    Args:
        audio_energy: Audio energy values
        audio_times: Time points for audio
        mouth_opening: Mouth opening values
        mouth_times: Time points for mouth
    
    Returns:
        Correlation coefficient in [0, 1] (normalized)
    """
    if len(audio_energy) == 0 or len(mouth_opening) == 0:
        return 0.5  # Neutral score if no data
    
    # Interpolate to common time grid
    min_time = max(audio_times.min() if len(audio_times) > 0 else 0, 
                   mouth_times.min() if len(mouth_times) > 0 else 0)
    max_time = min(audio_times.max() if len(audio_times) > 0 else 1, 
                   mouth_times.max() if len(mouth_times) > 0 else 1)
    
    if max_time <= min_time:
        return 0.5
    
    # Create common time grid
    common_times = np.linspace(min_time, max_time, num=min(100, max(len(audio_energy), len(mouth_opening))))
    
    # Interpolate both signals
    audio_interp = np.interp(common_times, audio_times, audio_energy)
    mouth_interp = np.interp(common_times, mouth_times, mouth_opening)
    
    # Normalize
    if audio_interp.std() > 0:
        audio_interp = (audio_interp - audio_interp.mean()) / audio_interp.std()
    if mouth_interp.std() > 0:
        mouth_interp = (mouth_interp - mouth_interp.mean()) / mouth_interp.std()
    
    # Compute correlation
    if len(audio_interp) > 1 and len(mouth_interp) > 1:
        correlation = np.corrcoef(audio_interp, mouth_interp)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            return 0.5
        
        # Normalize to [0, 1] where 1 = perfect sync, 0 = no sync
        # Correlation ranges from -1 to 1, so we map: -1 -> 0, 1 -> 1
        sync_score = (correlation + 1) / 2.0
        
        return float(np.clip(sync_score, 0.0, 1.0))
    
    return 0.5


def compute_audio_sync_score(audio_path: str, landmarks: Dict, video_fps: float = 1.0, 
                            window_size: float = 0.3, hop_size: float = 0.1) -> Tuple[float, Dict]:
    """
    Compute audio-visual lip-sync score with detailed analysis.
    
    Args:
        audio_path: Path to audio file (WAV)
        landmarks: Landmarks dictionary with mouth information
        video_fps: Frames per second of the video
        window_size: Audio window size in seconds
        hop_size: Audio hop size in seconds
    
    Returns:
        Tuple of (sync_score, analysis_dict) where:
        - sync_score: Overall sync score [0, 1] (1.0 = perfect, 0.0 = no sync)
        - analysis_dict: Contains frame-level sync scores and mismatch regions
    """
    try:
        # Extract audio features with improved method
        audio_times, audio_energy = extract_audio_features(audio_path, window_size, hop_size)
        
        # Also extract MFCC features for better accuracy (improves sync detection)
        combined_audio = audio_energy
        if LIBROSA_AVAILABLE and len(audio_energy) > 0:
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                frame_length = int(window_size * sr)
                hop_length = int(hop_size * sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
                # Use first MFCC coefficient (energy-related) as additional feature
                mfcc_energy = mfcc[0, :]
                # Normalize
                if mfcc_energy.max() > mfcc_energy.min():
                    mfcc_energy = (mfcc_energy - mfcc_energy.min()) / (mfcc_energy.max() - mfcc_energy.min())
                # Align lengths
                min_len = min(len(audio_energy), len(mfcc_energy))
                # Combine RMS and MFCC energy (weighted combination for better accuracy)
                # MFCC captures spectral characteristics that correlate better with speech
                combined_audio = 0.7 * audio_energy[:min_len] + 0.3 * mfcc_energy[:min_len]
                # Update audio_times to match
                audio_times = audio_times[:min_len]
            except Exception as e:
                print(f"Warning: Could not extract MFCC features, using RMS only: {e}")
                # combined_audio already set to audio_energy above
        
        # Extract mouth opening timeline
        mouth_times, mouth_opening = extract_mouth_opening_timeline(landmarks, video_fps)
        
        if len(combined_audio) == 0 or len(mouth_opening) == 0:
            print("Warning: No audio or mouth data available for sync analysis")
            return 0.5, {"frame_sync_scores": [], "mismatch_regions": []}
        
        # Compute overall correlation
        sync_score = compute_correlation(combined_audio, mouth_opening, audio_times, mouth_times)
        
        # Compute frame-level sync scores for detailed analysis
        frame_sync_scores = []
        mismatch_regions = []
        
        # Align audio and video timelines
        min_time = max(audio_times.min() if len(audio_times) > 0 else 0,
                       mouth_times.min() if len(mouth_times) > 0 else 0)
        max_time = min(audio_times.max() if len(audio_times) > 0 else 1,
                       mouth_times.max() if len(mouth_times) > 0 else 1)
        
        if max_time > min_time:
            # Create frame-level analysis
            num_frames = len(landmarks.get("frames", []))
            for frame_idx in range(num_frames):
                frame_time = frame_idx / video_fps
                
                # Find nearest audio window
                audio_idx = np.argmin(np.abs(audio_times - frame_time))
                if audio_idx < len(combined_audio):
                    # Get local correlation around this frame
                    window_size_frames = max(3, int(0.5 * video_fps))  # 0.5 second window
                    start_idx = max(0, frame_idx - window_size_frames // 2)
                    end_idx = min(num_frames, frame_idx + window_size_frames // 2)
                    
                    if end_idx > start_idx:
                        local_mouth = mouth_opening[start_idx:end_idx]
                        local_mouth_times = mouth_times[start_idx:end_idx]
                        
                        # Find corresponding audio segment
                        audio_start_idx = max(0, audio_idx - window_size_frames // 2)
                        audio_end_idx = min(len(combined_audio), audio_idx + window_size_frames // 2)
                        local_audio = combined_audio[audio_start_idx:audio_end_idx]
                        local_audio_times = audio_times[audio_start_idx:audio_end_idx]
                        
                        if len(local_audio) > 1 and len(local_mouth) > 1:
                            # Compute local correlation
                            local_sync = compute_correlation(
                                local_audio, local_mouth, 
                                local_audio_times, local_mouth_times
                            )
                            frame_sync_scores.append({
                                "frame_idx": frame_idx,
                                "time": frame_time,
                                "sync_score": local_sync
                            })
                            
                            # Identify mismatch regions (sync < 0.4)
                            if local_sync < 0.4:
                                mismatch_regions.append({
                                    "start_time": frame_time,
                                    "end_time": min(frame_time + 1.0, max_time),
                                    "sync_score": local_sync,
                                    "severity": "high" if local_sync < 0.2 else "moderate"
                                })
        
        analysis = {
            "frame_sync_scores": frame_sync_scores,
            "mismatch_regions": mismatch_regions,
            "overall_sync": sync_score
        }
        
        return sync_score, analysis
    
    except Exception as e:
        print(f"Error computing audio sync score: {e}")
        return 0.5, {"frame_sync_scores": [], "mismatch_regions": []}


def load_landmarks(landmarks_path: str) -> Dict:
    """Load landmarks from JSON file."""
    with open(landmarks_path, 'r') as f:
        return json.load(f)


def format_time_range(start_sec: float, end_sec: float) -> str:
    """Format time range as MM:SS-MM:SS."""
    start_min = int(start_sec // 60)
    start_sec_remainder = int(start_sec % 60)
    end_min = int(end_sec // 60)
    end_sec_remainder = int(end_sec % 60)
    return f"{start_min:02d}:{start_sec_remainder:02d}–{end_min:02d}:{end_sec_remainder:02d}"

