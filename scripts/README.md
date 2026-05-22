# Video Brand Detective 🕵️‍♂️📹

An automated AI-powered pipeline to meticulously scan ANY video and track every brand appearance, product placement, and audio mention. It uses `yt-dlp`, `ffmpeg`, and Google's Gemini models to exhaustively document brand exposures, outputting detailed timestamps, Exact on-screen locations, audio transcriptions, and more!

## Requirements
- Python 3.8+
- `ffmpeg` installed on your system
- A Google Gemini API Key

## Setup
1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Google API Key:
   Create a `.env` file in the same directory and add:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
   Or create a `google_api_key.txt` file containing only your API key.

## Usage
Simply run the wrapper script with any YouTube URL or local video path:
```bash
python analyze_video.py <video_url_or_local_path>
```

**Example:**
```bash
python analyze_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Output
The script automatically chunks the video, processes it, and generates two reports in the `universal_reports/` folder:
1. **Summary Report (`...summary...csv`)**: High-level aggregation showing unique visual and audio exposures for each brand.
2. **Detailed Log (`...detailed...csv`)**: Chronological listing of *every single appearance*, including:
   - Exata Location on Screen
   - Audio Transcription
   - Formats (Overlay, Clothing, Product, etc.)
   - Timestamps and clickable direct links
