# YouTube Semantic Segmenter

A Python tool that downloads YouTube videos, transcribes them using OpenAI's Whisper model, and segments them into semantically meaningful chunks with transcripts.

## Features

-  Downloads YouTube videos using yt-dlp
-  Transcribes audio using OpenAI's Whisper model (supports both base and large-v3)
-  Creates semantically meaningful chunks based on content
-  Validates chunk quality and audio processing
-  Provides detailed transcripts for each chunk
-  Parallel processing for efficient chunk generation
-  Audio preprocessing for better quality
-  Detailed metrics and metadata for each processing step

## Requirements

-  Python 3.8+
-  FFmpeg
-  yt-dlp
-  Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/youtube-semantic-segmenter.git
cd youtube-semantic-segmenter
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install FFmpeg if not already installed:

-  On macOS: `brew install ffmpeg`
-  On Ubuntu: `sudo apt-get install ffmpeg`
-  On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

Basic usage:

```bash
python cli.py https://www.youtube.com/watch?v=VIDEO_ID
```

With additional options:

```bash
python cli.py https://www.youtube.com/watch?v=VIDEO_ID --model-size base --preprocess-audio --validate-chunks --collect-metrics -v
```

### Options

-  `--model-size`: Choose between 'base' and 'large-v3' Whisper models (default: large-v3)
-  `--preprocess-audio`: Apply audio preprocessing for better quality
-  `--validate-chunks`: Enable chunk quality validation
-  `--collect-metrics`: Save processing metrics
-  `-v` or `--verbose`: Enable verbose logging

## Output

The tool generates the following outputs in the `output` directory:

1. `chunks/`: Directory containing WAV files for each segment
2. `metadata.json`: Detailed information about each chunk including:
   -  Timestamps
   -  Transcripts
   -  Quality metrics
   -  Confidence scores
3. `metrics.json`: Overall processing metrics

## Viewing Transcripts

To view transcripts for each chunk, check the `metadata.json` file in the output directory. Each chunk entry contains:

-  Start and end times
-  Full transcript text
-  Quality scores
-  Confidence metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
