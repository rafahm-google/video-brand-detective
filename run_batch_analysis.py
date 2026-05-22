import sys
import os
import subprocess
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [BATCH-RUNNER] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Google Drive destination folder (pointing to Industry Sizing subfolder where write is supported)
DRIVE_DEST_DIR = Path("/usr/local/google/home/rafahm/DriveFileStream/.shortcut-targets-by-id/1y2NF1pSgaNci9qK37esVQj42IjxVmYqI/Industry Sizing")

def run_analysis_for_video_id(video_id: str):
    # Check if video_id is actually a local file path
    if Path(video_id).exists() or video_id.startswith("/"):
        target = video_id
        logging.info(f"=== Processing Local Video Path: {target} ===")
    else:
        target = f"https://www.youtube.com/watch?v={video_id}"
        logging.info(f"=== Processing Video ID: {video_id} ===")
        logging.info(f"URL: {target}")

    # Running the analyze_video.py CLI inside venv
    cmd = ["venv/bin/python3", "scripts/analyze_video.py", target]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Successfully ran analysis for: {video_id}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running analysis for: {video_id}: {e}")
        return False

def copy_reports_to_drive():
    # Reports are saved in universal_reports
    local_reports_dir = Path("universal_reports")
    if not local_reports_dir.exists():
        logging.warning("No local universal_reports directory found.")
        return

    if not DRIVE_DEST_DIR.exists():
        logging.error(f"Destination Google Drive folder does not exist: {DRIVE_DEST_DIR}")
        return

    logging.info(f"Copying and renaming reports to Google Drive folder: {DRIVE_DEST_DIR}")
    copied_count = 0
    for file in local_reports_dir.glob("*.csv"):
        filename = file.name
        new_filename = filename
        
        if "_detailed_" in filename:
            # e.g. 2025-07-27_detailed_e2laV-0pImg.csv -> 2025-07-27_e2laV-0pImg_logs.csv
            parts = filename.split("_detailed_")
            if len(parts) == 2:
                new_filename = f"{parts[0]}_{parts[1].replace('.csv', '')}_logs.csv"
        elif "_summary_" in filename:
            # e.g. 2025-07-27_summary_e2laV-0pImg.csv -> 2025-07-27_e2laV-0pImg_summary.csv
            parts = filename.split("_summary_")
            if len(parts) == 2:
                new_filename = f"{parts[0]}_{parts[1].replace('.csv', '')}_summary.csv"
                
        dest_file = DRIVE_DEST_DIR / new_filename
        try:
            shutil.copyfile(file, dest_file)
            logging.info(f"Copied & Renamed: {filename} -> {new_filename}")
            copied_count += 1
        except Exception as e:
            logging.error(f"Failed to copy {filename} to Drive: {e}")
            
    logging.info(f"Finished copying {copied_count} reports to Google Drive.")

# Shared Drive pre-downloaded videos directory
LOCAL_VIDEOS_DIR = Path("/usr/local/google/home/rafahm/DriveFileStream/Shared drives/[LCS BR] PB Futebol/videos")

def parse_video_ids_from_csv(csv_path: Path) -> list:
    import csv
    video_ids = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            
            try:
                content_idx = header.index("Content")
            except ValueError:
                content_idx = 0
                
            for row in reader:
                if not row or len(row) <= content_idx:
                    continue
                val = row[content_idx].strip()
                if val and val != "Total":
                    video_ids.append(val)
    except Exception as e:
        logging.error(f"Failed to parse CSV file {csv_path}: {e}")
    return video_ids

def find_local_video_file(video_id: str):
    if not LOCAL_VIDEOS_DIR.exists():
        return None
        
    expected_filename = f"video_{video_id}.mp4"
    expected_filepath = LOCAL_VIDEOS_DIR / expected_filename
    
    if expected_filepath.exists():
        return expected_filepath
        
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: venv/bin/python3 run_batch_analysis.py <video_id_or_csv_path_or_comma_list>")
        sys.exit(1)

    first_arg = sys.argv[1]
    video_ids = []
    
    # 1. Check if first argument is an existing CSV file
    if first_arg.endswith(".csv") and Path(first_arg).exists():
        logging.info(f"CSV input detected: {first_arg}")
        video_ids = parse_video_ids_from_csv(Path(first_arg))
        logging.info(f"Successfully loaded {len(video_ids)} video IDs from CSV.")
    else:
        # 2. Handle space-separated or single comma-separated arguments
        for arg in sys.argv[1:]:
            if "," in arg:
                video_ids.extend([vid.strip() for vid in arg.split(",") if vid.strip()])
            else:
                video_ids.append(arg.strip())

    # Filter out duplicate video IDs
    video_ids = list(dict.fromkeys(video_ids))
    logging.info(f"Starting batch analysis for {len(video_ids)} video(s)...")

    success_count = 0
    for idx, video_id in enumerate(video_ids):
        logging.info(f"--- Processing item {idx+1}/{len(video_ids)} ---")
        
        # Check if it is already an absolute local path
        if Path(video_id).exists() or video_id.startswith("/"):
            target = video_id
        else:
            # Check if this video exists in the pre-downloaded videos directory
            local_file = find_local_video_file(video_id)
            if local_file:
                target = str(local_file)
                logging.info(f"Found pre-downloaded local file for Video ID '{video_id}' at: {target}")
            else:
                target = video_id
                logging.warning(f"No pre-downloaded local file found for Video ID '{video_id}'. Passing target directly.")
                
        if run_analysis_for_video_id(target):
            success_count += 1

    logging.info(f"Batch analysis completed. Success: {success_count}/{len(video_ids)}")
    
    # Copy all generated reports to Drive
    copy_reports_to_drive()

if __name__ == "__main__":
    main()
