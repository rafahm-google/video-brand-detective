# run_offline_analysis.py
# This script is designed to be run in an environment without internet access,
# such as a cloudtop. It reads a metadata CSV file and analyzes pre-downloaded
# local video files.

import csv
import logging
import time
from typing import List, Dict
from pathlib import Path
import sys

# Import the new offline analysis entry point
from offline_analysis_pipeline import main_offline, cleanup_cached_media, force_clear_specific_files_from_trash

# --- CONFIGURATION ---
METADATA_CSV_PATH = Path("~/DriveFileStream/My Drive/Brasileirao Files/videos_metadata.csv").expanduser()
WAIT_TIME_SECONDS = 30 # Wait time between processing videos

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [OFFLINE-RUNNER] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_metadata_csv(path: Path) -> List[Dict]:
    """Reads the video metadata from the CSV file."""
    if not path.exists():
        logging.error(f"Metadata file not found: '{path}'. Cannot proceed.")
        return []

    videos = []
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                videos.append(row)
        logging.info(f"Found {len(videos)} videos in metadata file.")
        return videos
    except Exception as e:
        logging.error(f"Failed to read or parse metadata CSV: {e}")
        return []

if __name__ == "__main__":
    videos_to_process = read_metadata_csv(METADATA_CSV_PATH)

    if not videos_to_process:
        logging.info("No videos found in metadata file. Exiting.")
        sys.exit(0)

    total_videos = len(videos_to_process)
    logging.info(f"--- Found {total_videos} videos to analyze from '{METADATA_CSV_PATH.name}' ---")

    for i, video_metadata in enumerate(videos_to_process, 1):
        title = video_metadata.get('title', 'No Title')
        video_id = video_metadata.get('video_id', 'N/A')
        local_path = Path("~/DriveFileStream/My Drive/Brasileirao Files/videos").expanduser() / video_metadata.get('local_path', '')
        video_metadata['local_path'] = local_path

        logging.info("="*80)
        logging.info(f"STARTING OFFLINE ANALYSIS FOR VIDEO {i}/{total_videos}: {title}")
        logging.info(f"Video ID: {video_id} | Local Path: {local_path}")
        logging.info("="*80)
        
        # Check if the local video file actually exists before starting
        if not Path(local_path).exists():
            logging.error(f"Video file not found for '{title}' at path '{local_path}'. SKIPPING.")
            continue

        # Call the main entry point of the offline pipeline
        success, returned_video_id = main_offline(video_metadata)

        if success:
            logging.info(f"SUCCESSFULLY completed analysis for: {title}")
        else:
            logging.error(f"FAILED analysis for: {title}. Check logs for details.")

        # Cleanup is still relevant for the temporary chunk files
        if returned_video_id:
            cleanup_cached_media(returned_video_id)
            force_clear_specific_files_from_trash(returned_video_id)
        else:
            logging.warning(f"Could not get a video_id for {title}, cannot perform media cleanup.")

        if i < total_videos:
            logging.info(f"Waiting for {WAIT_TIME_SECONDS} seconds before starting the next video...")
            time.sleep(WAIT_TIME_SECONDS)

    logging.info("="*80)
    logging.info("ALL VIDEOS FROM METADATA FILE HAVE BEEN PROCESSED.")
    logging.info("="*80)
