# generate_metadata.py
# This script is intended to be run on a machine with access to YouTube (e.g., in Brazil).
# It identifies the target videos, fetches their metadata using yt-dlp, and saves it
# to a CSV file. This file, along with the downloaded videos, can then be moved to
# another environment (like a cloudtop) for offline analysis.

import csv
import logging
from typing import List, Dict
from pathlib import Path

# Import functions from the existing scripts
from run_all_videos import get_and_filter_videos, CHANNEL_URL, KEYWORDS_IN_TITLE, MAX_VIDEOS_TO_CHECK
from analysis_pipeline import get_video_metadata

# --- CONFIGURATION ---
METADATA_CSV_PATH = Path("videos_metadata.csv")

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [METADATA-GEN] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_metadata_file(videos: List[Dict[str, str]], output_path: Path):
    """
    Fetches detailed metadata for a list of video URLs and saves it to a CSV file.
    """
    if not videos:
        logging.warning("Video list is empty. No metadata file will be generated.")
        return

    header = ['video_id', 'title', 'url', 'upload_datetime', 'duration_seconds', 'local_path']
    rows_written = 0

    logging.info(f"Generating metadata for {len(videos)} videos. This may take a while...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, video_info in enumerate(videos, 1):
            url = video_info['url']
            title = video_info['title']
            logging.info(f"Fetching metadata for video {i}/{len(videos)}: {title}")

            metadata = get_video_metadata(url)
            
            if metadata:
                video_id, video_title, duration, dt = metadata
                
                # The expected local path for the video file
                local_path = f"video_{video_id}.mp4"

                writer.writerow([
                    video_id,
                    video_title,
                    url,
                    dt.isoformat(),
                    duration,
                    local_path
                ])
                rows_written += 1
            else:
                logging.error(f"Could not fetch metadata for '{title}' ({url}). It will be skipped.")

    logging.info(f"Successfully generated metadata for {rows_written} videos.")
    logging.info(f"Metadata file saved to: '{output_path}'")


if __name__ == "__main__":
    logging.info("Starting metadata generation process...")
    
    # 1. Find all relevant videos from the channel
    videos_to_process = get_and_filter_videos(CHANNEL_URL, KEYWORDS_IN_TITLE, MAX_VIDEOS_TO_CHECK)

    # 2. Generate the metadata CSV file
    if videos_to_process:
        generate_metadata_file(videos_to_process, METADATA_CSV_PATH)
    else:
        logging.info("No videos found matching the criteria. Exiting.")

    logging.info("="*80)
    logging.info("METADATA GENERATION COMPLETE.")
    logging.info(f"The next step is to upload the '{METADATA_CSV_PATH.name}' file and all 'video_*.mp4' files to your cloud environment.")
    logging.info("="*80)
