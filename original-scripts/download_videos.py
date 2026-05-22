# download_videos.py
# This script finds relevant videos from a YouTube channel and downloads them locally.
# It is designed to be run before the analysis pipeline to ensure all videos are
# available locally, avoiding potential cookie expiration issues during a long-running analysis.

import subprocess
import json
import logging
import time
from typing import List, Dict
import sys
from pathlib import Path

# Import configuration and metadata functions from the existing analysis pipeline
from analysis_pipeline import Config, get_video_metadata, sanitize_filename

# --- CONFIGURATION FOR THE DOWNLOADER ---
# These settings are copied from run_all_videos.py to ensure we download the correct videos.
CHANNEL_URL = "https://www.youtube.com/@CazeTV/streams"
KEYWORDS_IN_TITLE = [
    #"PAULISTÃO 2025",
    "BRASILEIRÃO 2025"
    #"COPA DO MUNDO DE CLUBES DA FIFA"
]
MAX_VIDEOS_TO_CHECK = 1000

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [DOWNLOADER] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_and_filter_videos(channel_url: str, keywords: List[str], max_videos: int) -> List[Dict[str, str]]:
    """
    Fetches video metadata using yt-dlp and then filters the results inside Python
    to select only completed live streams that match the keywords.
    (This function is adapted from run_all_videos.py)
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
    else:
        logging.warning("Cookie file not found. Trying to use cookies from browser. This may fail if you are not logged in.")
        command.extend(['--cookies-from-browser', 'chrome'])

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
        logging.error(f"The 'yt-dlp' command failed. This may be due to an authentication issue (invalid cookies).")
        logging.error(f"Error details: {e.stderr.decode(errors='ignore') if e.stderr else e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching videos: {e}")
        return []

def download_video(video_url: str, video_id: str) -> bool:
    """
    Downloads a single video to a local file, if it doesn't already exist.
    """
    local_video_path = Path(f"video_{video_id}.mp4")
    if local_video_path.exists():
        logging.info(f"Video '{local_video_path}' already exists. Skipping download.")
        return True

    logging.info(f"Downloading video to '{local_video_path}'...")
    
    python_executable_path = Path(sys.executable)
    yt_dlp_executable_path = python_executable_path.parent / 'yt-dlp'

    yt_dlp_common = [str(yt_dlp_executable_path), '--no-check-certificate', '--retries', 'infinite', '--quiet']
    
    if Config.COOKIE_FILE_PATH and Config.COOKIE_FILE_PATH.exists():
        yt_dlp_common.extend(['--cookies', str(Config.COOKIE_FILE_PATH)])
    else:
        yt_dlp_common.extend(['--cookies-from-browser', 'chrome'])
    
    try:
        quality = Config.STREAM_QUALITY.replace("p", "")
        format_string = f'bestvideo[height<={quality}]+bestaudio/best'
        cmd = yt_dlp_common + ['-f', format_string, '--merge-output-format', 'mp4', '-o', str(local_video_path), video_url]
        
        # Using subprocess.run and capturing output for better error logging
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

        logging.info(f"Successfully downloaded {video_url}")
        return True
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip()
        logging.error(f"A command failed during video download for {video_url}.")
        logging.error(f"Stderr: {error_output}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during video download for {video_url}: {e}")
        return False

if __name__ == "__main__":
    videos_to_process = get_and_filter_videos(CHANNEL_URL, KEYWORDS_IN_TITLE, MAX_VIDEOS_TO_CHECK)

    if not videos_to_process:
        logging.info("No videos found matching the criteria. Exiting.")
    else:
        total_videos = len(videos_to_process)
        logging.info(f"--- Found {total_videos} videos to download ---")

        for i, video_info in enumerate(videos_to_process, 1):
            url = video_info['url']
            title = video_info['title']
            
            logging.info("="*80)
            logging.info(f"Processing video {i}/{total_videos}: {title}")

            # We need the video_id to name the file correctly.
            # This also serves as a check to see if the video is accessible.
            metadata = get_video_metadata(url)
            if not metadata:
                logging.error(f"Could not get metadata for '{title}' ({url}). Skipping download.")
                continue
            
            video_id, _, _, _ = metadata
            download_success = download_video(url, video_id)

            if not download_success:
                logging.error(f"Failed to download video: {title}. Check logs for details.")

        logging.info("="*80)
        logging.info("ALL VIDEO DOWNLOADS ATTEMPTED.")
        logging.info("="*80)
