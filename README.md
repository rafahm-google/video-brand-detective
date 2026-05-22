# Video Brand Detective

An automated, parallelized multimodal AI vision pipeline powered by Google's Gemini API (`gemini-3-flash-preview`) to detect and verify brand placements and verbal insertions inside video and audio streams.

## Features
* **Multimodal Scanning**: Integrates parallel video frame sampling and audio track listening in a single pipeline using Google GenAI.
* **Batch Verification Layer**: Features a 5-snapshot spatial-temporal verification layer to validate candidate visual detections, dramatically reducing false positives and timing errors.
* **Fault Tolerance**: Built-in exponential backoff retry handling for API rate limits and robust error isolation.
* **Deduplication**: Post-processing deduplication using sliding time windows to group contiguous brand exposures.

## Requirements

### System Dependencies
The pipeline requires `ffmpeg` and `yt-dlp` for audio extraction and media segmenting:
* **Debian/Ubuntu/gLinux**: `sudo apt install ffmpeg yt-dlp`
* **Mac (Homebrew)**: `brew install ffmpeg yt-dlp`

### Python Dependencies
Make sure you have Python 3.10+ installed, then run:
```bash
pip install -r scripts/requirements.txt
```

## Setup
Create a `.env` file in the project root and add your Google Gemini API Key:
```env
GOOGLE_API_KEY="your-gemini-api-key"
```

## How to Run
The core brand detective engine is executed via `scripts/analyze_video.py`:

```bash
# Analyze a YouTube video
python scripts/analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Analyze a local video file directly
python scripts/analyze_video.py "/path/to/local_video.mp4"
```

Reports (Summary and Detailed Log CSVs) will be automatically generated inside the `universal_reports/` folder in the project root directory.
