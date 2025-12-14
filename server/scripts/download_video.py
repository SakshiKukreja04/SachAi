#!/usr/bin/env python3
"""
Python wrapper script to download videos using yt-dlp.
This avoids Node.js shell issues on Windows.
"""
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 4:
        print("Usage: download_video.py <python_exe> <output_path> <video_url> [yt_dlp_args...]")
        sys.exit(1)
    
    python_exe = sys.argv[1]
    output_path = Path(sys.argv[2])
    video_url = sys.argv[3]
    yt_dlp_args = sys.argv[4:] if len(sys.argv) > 4 else []
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build yt-dlp command
    cmd = [
        python_exe,
        '-m', 'yt_dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '--no-playlist',
        '--quiet',
        '--no-warnings',
        '-o', str(output_path),
        video_url
    ]
    
    # Add any additional args
    cmd.extend(yt_dlp_args)
    
    try:
        # Use subprocess.run which handles paths with spaces correctly
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
        )
        
        if result.returncode == 0:
            if output_path.exists():
                print(f"SUCCESS: {output_path}")
                sys.exit(0)
            else:
                print(f"ERROR: Download completed but file not found: {output_path}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                sys.exit(1)
        else:
            print(f"ERROR: yt-dlp failed with code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            sys.exit(result.returncode)
            
    except Exception as e:
        print(f"ERROR: Exception during download: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

