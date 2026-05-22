# run_all_videos.py
# This script finds relevant videos from a YouTube channel and runs the analysis pipeline for each one.
# VERSION 4.3 - Adds logic to skip videos if their report already exists.

import subprocess
import json
import logging
import time
from typing import List, Dict
import sys
from pathlib import Path

# Import all necessary functions for the script
from analysis_pipeline import (
    main as run_single_video_analysis, 
    cleanup_cached_media, 
    force_clear_specific_files_from_trash,
    get_video_metadata,  # <--- NEWLY IMPORTED for the check
    sanitize_filename,   # <--- NEWLY IMPORTED for the check
    Config
)


# --- CONFIGURATION FOR THE RUNNER ---
CHANNEL_URL = "https://www.youtube.com/@CazeTV/streams"
KEYWORDS_IN_TITLE = [
    "PAULISTÃO 2025",
    "PAULISTÃO FEMININO 2025",
    "BRASILEIRÃO 2025",
    "COPINHA 2025"
    #"COPA DO MUNDO DE CLUBES DA FIFA"
]
MAX_VIDEOS_TO_CHECK = 10000
WAIT_TIME_SECONDS = 30 # Defined wait time here

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [RUNNER] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_and_filter_videos(channel_url: str, keywords: List[str], max_videos: int) -> List[Dict[str, str]]:
    """
    Fetches video metadata using yt-dlp and then filters the results inside Python
    to select only completed live streams that match the keywords.
    """
    logging.info(f"Fetching metadata for the last {max_videos} videos from {channel_url}...")
    
    python_executable_path = Path(sys.executable)
    yt_dlp_executable_path = python_executable_path.parent / 'yt-dlp'

    if not yt_dlp_executable_path.exists():
        logging.error(f"FATAL: Could not find 'yt-dlp' at: {yt_dlp_executable_path}")
        return []

    command = [
        str(yt_dlp_executable_path),
        '--no-check-certificate',
        '--flat-playlist',
        '--dump-json',
        '--playlist-end', str(max_videos),
        channel_url
    ]

    if Config.COOKIE_FILE_PATH and Config.COOKIE_FILE_PATH.exists():
        command.extend(['--cookies', str(Config.COOKIE_FILE_PATH)])

    try:
        result = subprocess.check_output(command, text=True, encoding='utf-8')
        video_lines = result.strip().split('\n')
        all_videos = [json.loads(line) for line in video_lines if line]

        logging.info("Filtering results in Python...")
        filtered_videos = []
        for video in all_videos:
            title = video.get('title', 'No Title')
            live_status = video.get('live_status')

            if live_status == 'was_live':
                if any(keyword.upper() in title.upper() for keyword in keywords):
                    logging.info(f"MATCH FOUND: '{title}'")
                    filtered_videos.append({'title': title, 'url': video.get('url')})
                else:
                    logging.info(f"Skipping (keyword not in title): '{title}'")
            else:
                logging.info(f"Skipping (status is '{live_status}'): '{title}'")
        
        logging.info(f"Found {len(filtered_videos)} completed live streams matching the criteria after filtering.")
        return filtered_videos

    except subprocess.CalledProcessError as e:
        logging.error(f"The 'yt-dlp' command failed. The program may be blocked or corrupted.")
        logging.error(f"Error details: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching videos: {e}")
        return []


if __name__ == "__main__":
    videos_to_process = get_and_filter_videos(CHANNEL_URL, KEYWORDS_IN_TITLE, MAX_VIDEOS_TO_CHECK)

    if not videos_to_process:
        logging.info("No videos found matching the criteria. Exiting.")
    else:
        total_videos = len(videos_to_process)
        logging.info(f"--- Found {total_videos} videos to analyze ---")
        for i, video in enumerate(videos_to_process, 1):
            logging.info(f"  {i}. {video['title']}")

        proceed = input("\nDo you want to start the analysis for all videos? (y/n): ")
        if proceed.lower() != 'y':
            logging.info("Analysis cancelled by user.")
            exit()

        for i, video_info in enumerate(videos_to_process, 1):
            title = video_info['title']
            url = video_info['url']

            logging.info("="*80)
            logging.info(f"Checking status for video {i}/{total_videos}: {title}")
            
            # --- NEW: LOGIC TO SKIP ALREADY PROCESSED VIDEOS ---
            # Step 1: Quickly fetch metadata to determine the expected report filename.
            metadata = get_video_metadata(url)
            if metadata:
                # Step 2: Construct the exact filename that would be created on success.
                _, video_title, _, video_upload_datetime = metadata
                sanitized_title = sanitize_filename(video_title)
                date_prefix = video_upload_datetime.strftime('%Y-%m-%d')
                expected_report_path = Path("output_reports_v2") / f"{date_prefix}_summary_report_{sanitized_title}.csv"

                # Step 3: Check if the report file already exists.
                if expected_report_path.exists():
                    logging.info(f"Report '{expected_report_path.name}' already exists. SKIPPING video.")
                    continue # Immediately move to the next video in the loop
            else:
                logging.error("Could not fetch metadata to check for existing report. Will attempt full analysis.")
            # --- END OF SKIP LOGIC ---

            logging.info(f"STARTING ANALYSIS FOR VIDEO {i}/{total_videos}: {title}")
            logging.info(f"URL: {url}")
            logging.info("="*80)

            # This part remains the same.
            success, video_id = run_single_video_analysis(url)

            if success:
                logging.info(f"SUCCESSFULLY completed analysis for: {title}")
            else:
                logging.error(f"FAILED analysis for: {title}. Check logs for details.")

            if video_id:
                cleanup_cached_media(video_id)
                force_clear_specific_files_from_trash(video_id)
            else:
                logging.warning(f"Could not get a video_id for {title}, cannot perform media cleanup.")

            if i < total_videos:
                logging.info(f"Waiting for {WAIT_TIME_SECONDS} seconds before starting the next video...")
                time.sleep(WAIT_TIME_SECONDS)

        logging.info("="*80)
        logging.info("ALL VIDEOS HAVE BEEN PROCESSED.")
        logging.info("="*80)