
# Automated Advertising Analysis for Football Broadcasts

This project contains a set of Python scripts designed to automatically analyze YouTube videos of football matches, identify all brand advertisements and mentions, and generate detailed reports.

It's built to be efficient and cost-effective, using AI to "watch" the game and log every ad, saving hundreds of hours of manual work.

## What it Does: The Big Picture

Imagine you want to know every single time a brand like *Coca-Cola* or *Sportingbet* appeared or was mentioned during a 2-hour football game on YouTube. This tool does exactly that, automatically.

It processes a list of videos and, for each one, produces two main reports:
1.  A **Detailed Log** of every single ad, with a timestamp and a direct link to that moment in the video.
2.  A **High-Level Summary Report** that provides strategic insights into each brand's advertising strategy, frequency, and messaging.

## How it Works: Step-by-Step

The process is managed by two main scripts that work together. Here's the journey from a YouTube URL to a final report:

1.  **Find the Games (`run_all_videos.py`)**: The process starts by scanning the CazéTV YouTube channel for recent, completed broadcasts that match keywords like "BRASILEIRÃO" or "PAULISTÃO". It filters out any upcoming or currently live streams.

2.  **Download & Prepare (`analysis_pipeline.py`)**: For a single game, the system downloads a 360p copy of the video and audio. It then slices the full game into smaller, 5-minute "chunks". This is like breaking a long movie into smaller scenes to make it easier to analyze.

3.  **Analyze in Parallel (`analysis_pipeline.py`)**: This is the core AI step. The system takes multiple 5-minute chunks and sends them to the Google Gemini AI for analysis *at the same time*. For each chunk, the AI:
    *   Looks at sampled video frames to identify visual logos, banners, and on-screen graphics.
    *   Listens to the audio to detect any spoken brand mentions by the commentators.
    *   Returns a structured list of everything it found.

4.  **Generate Reports (`analysis_pipeline.py`)**: Once all the chunks from a video have been analyzed, the system gathers all the individual findings.
    *   It first creates the **Detailed Log**, a chronological list of every ad.
    *   Then, using this data, it makes a single, efficient API call to have the AI generate a **Strategic Summary** for each brand, analyzing their overall campaign within that game.

5.  **Clean Up & Repeat (`run_all_videos.py`)**: After the reports for one video are successfully saved, the system deletes the large downloaded video and audio files to save disk space. It then moves on to the next video in the list and repeats the process.

## The Components: A Deeper Look

The project is split into two key files for clarity and function.

### `run_all_videos.py` - The Orchestrator

Think of this script as the **Project Manager**. It doesn't do the analysis itself, but it manages the entire workflow from start to finish.

-   **`get_and_filter_videos()`**: Its first job is to act like a scout. It goes to the specified YouTube channel and finds all the relevant, completed games based on the configured keywords.
-   **Main Loop**: It then iterates through the list of games, one by one. For each game, it tells the "Analysis Engine" to start its work and waits for it to finish.
-   **`cleanup_cached_media()`**: After the engine is done with a video, this orchestrator cleans up the large downloaded files before starting the next video.

### `analysis_pipeline.py` - The Analysis Engine

This script is the **Hard Worker**. It's designed to do one job perfectly: analyze a single video and produce reports for it.

-   **Configuration (`Config` class)**: This is the settings panel at the top of the file. It controls things like video quality, how many tasks to run in parallel (`NUM_WORKERS`), and which brands to ignore (like "FIFA" or "CazéTV" itself).
-   **Media Preparation (`get_video_metadata`, `download_and_split_media`)**: Gets all the necessary information about the video (like its title and date) and handles the downloading and chunking process.
-   **Core AI Analysis (`analyze_chunk`)**: This is the most important function. It takes a 5-minute chunk of video and audio, prepares it for the AI, sends it to Google with a very specific set of instructions (the "prompt"), and carefully parses the results. The retry logic here is crucial for handling API rate limits.
-   **Reporting (`create_summary_report`, `create_detailed_report`)**: These functions take the raw data from all the analyzed chunks and format it into the final, human-readable CSV reports. The `generate_all_summaries_in_batch` function is a key optimization here, as it gets all the AI-written summaries in one go, saving significant time and money.

## How to Use

1.  **Prerequisites**: Make sure you have Python 3 installed. Then, install the required libraries using the command:
    ```bash
    pip install -r requirements.txt
    ```
    *(You would need to create a `requirements.txt` file containing `google-generativeai`, `pandas`, `opencv-python`, `Pillow`, `tqdm`, etc.)*

2.  **Configuration**:
    *   **API Key**: Place your Google AI API key in a file named `google_api_key.txt` in the same directory.
    *   **Cookies (Optional)**: If you need to analyze age-restricted or member-only videos, place a valid `cookies.txt` file in the directory.
    *   **Keywords**: Open `run_all_videos.py` and adjust the `KEYWORDS_IN_TITLE` list to find the videos you're interested in.

3.  **Run the Analysis**:
    Execute the orchestrator script from your terminal:
    ```bash
    python run_all_videos.py
    ```
    The script will show you the list of videos it found and ask for your confirmation before starting the analysis.

## Expected Output

After running, you will find your reports inside a new folder named `output_reports/`. For each video processed, two files will be generated:

1.  **`YYYY-MM-DD_detailed_log_[video_title].csv`**: The granular log.
    > Contains one row for every single ad detection, with columns like `sponsor`, `Horário no Vídeo`, `Tipo de Detecção`, and a clickable `Link para o Momento`.

2.  **`YYYY-MM-DD_summary_report_[video_title].csv`**: The strategic summary.
    > Contains one row per brand, summarizing their entire activity with columns like `Frequência Visual`, `Menções em Áudio`, `Detalhamento dos Formatos`, and the AI-generated `Resumo Analítico da Estratégia`.