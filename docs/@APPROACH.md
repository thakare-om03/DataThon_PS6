# Technical Approach

This document outlines the technical approach used in the YouTube Semantic Segmenter.

## Core Components

1. **Video Download & Audio Extraction**

   -  Uses `yt-dlp` for reliable YouTube video downloading
   -  Extracts high-quality WAV audio at 16kHz mono
   -  Optional audio preprocessing using FFmpeg filters

2. **Audio Transcription**

   -  Utilizes OpenAI's Whisper model (supports base and large-v3)
   -  Implements word-level timestamps
   -  Handles model-specific features (e.g., word confidence for large-v3)
   -  Temperature control for consistent output

3. **Semantic Segmentation**

   -  TF-IDF vectorization for semantic similarity
   -  NLTK for sentence tokenization and completeness analysis
   -  Chunk duration constraints (min: 20s, max: 40s)
   -  Voice activity detection using webrtcvad

4. **Chunk Processing**
   -  Parallel processing with optimized worker count
   -  FFmpeg for precise audio extraction
   -  Quality validation metrics
   -  Metadata generation with transcripts

## Segmentation Algorithm

1. **Initial Segmentation**

   -  Process transcription segments sequentially
   -  Consider semantic similarity between segments
   -  Check for complete thoughts using NLTK
   -  Enforce duration constraints

2. **Boundary Refinement**

   -  Detect voice activity at chunk boundaries
   -  Adjust boundaries to align with speech
   -  Merge short segments when appropriate
   -  Validate chunk quality

3. **Quality Metrics**
   -  Duration scoring
   -  Confidence scoring
   -  Semantic completeness
   -  Audio quality assessment

## Performance Optimization

1. **Resource Management**

   -  Dynamic worker count based on system resources
   -  Memory-efficient audio processing
   -  Temporary file cleanup
   -  Progress tracking with tqdm

2. **Error Handling**
   -  Comprehensive logging
   -  Graceful fallbacks for unsupported features
   -  Clean error messages
   -  Automatic retries where appropriate

## Latest Updates

1. **Model Compatibility**

   -  Adaptive feature selection based on model capabilities
   -  Graceful handling of base vs large model differences
   -  Configurable model parameters

2. **Transcript Access**

   -  Detailed metadata JSON with transcripts
   -  Word-level timing information
   -  Confidence scores when available
   -  Easy access to chunk content

3. **Quality Improvements**
   -  Enhanced audio preprocessing
   -  Better semantic boundary detection
   -  Improved chunk validation
   -  More detailed metrics
