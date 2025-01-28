# Testing Guide

This document outlines how to test the YouTube Semantic Segmenter.

## Prerequisites

1. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Make sure FFmpeg is installed and accessible from command line:

```bash
ffmpeg -version
```

## Basic Testing

1. **Quick Test**

   ```bash
   python cli.py https://www.youtube.com/watch?v=-uleG_Vecis --model-size base
   ```

   This will process a short educational video using the base model.

2. **Check Outputs**
   -  Verify `output/chunks/` directory contains WAV files
   -  Check `output/metadata.json` for transcripts
   -  Review `output/metrics.json` for performance data

## Advanced Testing

1. **Test Different Model Sizes**

   ```bash
   # Test with base model
   python cli.py <URL> --model-size base

   # Test with large model
   python cli.py <URL> --model-size large-v3
   ```

2. **Test Audio Processing**

   ```bash
   # Test with preprocessing
   python cli.py <URL> --preprocess-audio

   # Test chunk validation
   python cli.py <URL> --validate-chunks
   ```

3. **Test with Different Video Types**
   -  Short videos (< 5 minutes)
   -  Medium videos (5-15 minutes)
   -  Long videos (> 15 minutes)
   -  Different languages
   -  Different audio qualities

## Viewing Results

1. **Check Transcripts**

   ```bash
   # Using Python
   import json
   with open('output/metadata.json') as f:
       metadata = json.load(f)

   # Print transcript for each chunk
   for chunk in metadata:
       print(f"Chunk {chunk['chunk_id']}:")
       print(f"Time: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
       print(f"Text: {chunk['text']}\n")
   ```

2. **Validate Audio Chunks**

   -  Use an audio player to check chunk quality
   -  Verify chunk boundaries align with speech
   -  Check for any audio artifacts

3. **Review Metrics**
   -  Check confidence scores
   -  Verify chunk durations
   -  Assess semantic completeness

## Automated Testing

Run the test suite:

```bash
python -m pytest test_segmenter.py -v
```

The test suite covers:

-  Video download functionality
-  Audio processing
-  Transcription accuracy
-  Chunk generation
-  Metadata creation

## Common Issues

1. **Missing Dependencies**

   -  Ensure FFmpeg is installed
   -  Check Python package versions
   -  Verify NLTK data is downloaded

2. **Audio Processing**

   -  Check FFmpeg installation
   -  Verify audio file permissions
   -  Monitor disk space

3. **Model Issues**
   -  Verify model compatibility
   -  Check available system memory
   -  Monitor GPU usage if available

## Performance Testing

1. **Resource Usage**

   ```bash
   # Test with different worker counts
   python cli.py <URL> --num-workers 2
   python cli.py <URL> --num-workers 4
   ```

2. **Processing Speed**
   -  Time processing of different video lengths
   -  Monitor memory usage
   -  Check CPU utilization

## Reporting Issues

When reporting issues:

1. Include full command used
2. Attach relevant logs
3. Provide system specifications
4. Share sample video URL if possible
