# analyze_video.py
# Universal Brand Analyzer - Entry Point

import sys
import logging
from pathlib import Path

# Ensure the directory is in path
sys.path.append(str(Path(__file__).parent))

from universal_pipeline import run_universal_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [UNIVERSAL-CLI] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <video_url_or_local_path>")
        sys.exit(1)
    
    target = sys.argv[1]
    logging.info(f"Starting universal brand analysis for: {target}")
    
    try:
        run_universal_analysis(target)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
