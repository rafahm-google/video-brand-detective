# Video Brand Analyzer

A universal brand detection tool that uses the Google Gemini API to identify brand logos and audio mentions in videos.

## Features
- **Visual Detection**: Identifies logos and graphics with screen coordinates.
- **Audio Recognition**: Transcribes and identifies brand mentions in audio.
- **Reporting**: Generates summary and detailed CSV reports with timestamps and context.
- **Flexible Input**: Supports local video files and YouTube URLs.

## Setup
1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

## Usage
Run the analysis by passing a video path or YouTube URL:
```bash
./venv/bin/python3 analyze_video.py "https://www.youtube.com/watch?v=aHxdMIukRGM"
```

The reports will be saved in the `universal_reports/` directory.

## Requirements
- Python 3.9+
- FFmpeg installed on your system.
- `yt-dlp` for YouTube downloads.
