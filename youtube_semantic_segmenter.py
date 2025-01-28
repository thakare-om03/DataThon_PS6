import os
import json
import torch
import whisper
import webrtcvad
import numpy as np
import logging
import multiprocessing
import psutil
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("semantic_segmenter.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class MetricsHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def emit(self, record):
        if hasattr(record, "metrics"):
            self.metrics.append(record.metrics)


class YoutubeSemanticSegmenter:
    def __init__(
        self,
        output_dir="output",
        num_workers: int = None,
        max_chunk_duration: float = 40.0,  # Increased for better semantic completeness
        min_chunk_duration: float = 20.0,  # Increased for meaningful content
        model_size: str = "large-v3",
        preprocess_audio: bool = True,
        validate_chunks: bool = True,
    ):
        """
        Initialize the segmenter with configurable parameters.

        Args:
            output_dir (str): Directory to store outputs
            num_workers (int, optional): Number of workers for parallel processing
            max_chunk_duration (float): Maximum duration of each chunk in seconds
            min_chunk_duration (float): Minimum duration of each chunk in seconds
            model_size (str): Whisper model size to use
            preprocess_audio (bool): Whether to apply audio preprocessing
            validate_chunks (bool): Whether to validate chunk quality
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.vad = webrtcvad.Vad(3)  # Maximum aggressiveness
        self.whisper_model = whisper.load_model(model_size)
        self.num_workers = self._optimize_worker_count(num_workers)
        self.max_chunk_duration = max_chunk_duration * 1000  # Convert to milliseconds
        self.min_chunk_duration = min_chunk_duration * 1000  # Convert to milliseconds
        self.preprocess_audio = preprocess_audio
        self.validate_chunks = validate_chunks
        self.metrics_handler = MetricsHandler()
        logging.getLogger().addHandler(self.metrics_handler)
        logger.info(f"Initialized with {self.num_workers} workers")

    def _optimize_worker_count(self, requested_workers: Optional[int] = None) -> int:
        """Determine optimal number of workers based on system resources"""
        available_memory = psutil.virtual_memory().available
        memory_based_workers = max(
            1, available_memory // (2 * 1024 * 1024 * 1024)
        )  # 2GB per worker
        cpu_based_workers = max(1, multiprocessing.cpu_count() - 1)
        default_workers = min(memory_based_workers, cpu_based_workers)
        return min(requested_workers or default_workers, default_workers)

    def preprocess_audio_file(self, audio_path: str) -> str:
        """Apply audio preprocessing for better quality"""
        if not self.preprocess_audio:
            return audio_path

        output_path = str(
            Path(audio_path).with_stem(Path(audio_path).stem + "_processed")
        )
        try:
            # Apply audio preprocessing filters
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-af",
                    "highpass=f=200,lowpass=f=3000,anlmdn=s=7:p=0.002:r=0.002",
                    "-ar",
                    "16000",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            return output_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"Audio preprocessing failed: {e.stderr.decode()}")
            return audio_path

    def download_video(self, youtube_url: str) -> str:
        """Download video and extract high-quality audio"""
        try:
            logger.info(f"Downloading video: {youtube_url}")
            output_template = str(self.output_dir / "%(title)s.%(ext)s")

            # Download with best audio quality
            subprocess.run(
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format",
                    "wav",
                    "--audio-quality",
                    "0",
                    "--postprocessor-args",
                    "-ar 16000 -ac 1",
                    "-o",
                    output_template,
                    youtube_url,
                ],
                check=True,
            )

            # Find the downloaded wav file
            wav_file = next(self.output_dir.glob("*.wav"), None)
            if not wav_file:
                raise FileNotFoundError("Failed to create WAV file")

            # Preprocess audio if enabled
            if self.preprocess_audio:
                wav_file = Path(self.preprocess_audio_file(str(wav_file)))

            logger.info(f"Audio saved to: {wav_file}")
            return str(wav_file)

        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper with enhanced settings"""
        try:
            logger.info("Starting audio transcription")

            # Base options that work with all models
            options = {
                "verbose": True,
                "word_timestamps": True,
                "condition_on_previous_text": True,
                "temperature": 0.0,  # Use greedy decoding for consistency
            }

            # Add advanced options only for large-v3 model
            if hasattr(self.whisper_model, "dims") and hasattr(
                self.whisper_model.dims, "n_vocab"
            ):
                if "large" in str(self.whisper_model.dims.n_vocab):
                    options["compute_word_confidence"] = True

            result = self.whisper_model.transcribe(audio_path, **options)
            logger.info("Transcription completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

    def detect_voice_activity(self, audio_data, sample_rate=16000):
        """Detect voice activity in audio frames"""
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        frames = self._frame_generator(frame_size, audio_data, sample_rate)

        voiced_frames = []
        for frame in frames:
            if self.vad.is_speech(frame.bytes, sample_rate):
                voiced_frames.append(frame)
        return voiced_frames

    def _frame_generator(self, frame_size: int, audio: np.ndarray, sample_rate: int):
        """Generate frames from audio data"""
        n = len(audio)
        offset = 0
        while offset + frame_size < n:
            yield Frame(
                audio[offset : offset + frame_size],
                sample_rate,
                timestamp=offset / sample_rate,
            )
            offset += frame_size

    def create_semantic_chunks(
        self, audio_path: str, transcription: Dict
    ) -> List[Dict]:
        """Create semantic chunks with improved context awareness"""
        logger.info("Creating semantic chunks...")

        # Initialize TF-IDF vectorizer for semantic similarity
        vectorizer = TfidfVectorizer(stop_words="english")

        def calculate_semantic_similarity(text1: str, text2: str) -> float:
            """Calculate semantic similarity between two text segments"""
            try:
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                return 0.0

        def is_complete_thought(text: str) -> bool:
            """Check if text forms a complete thought"""
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return False
            last_sentence = sentences[-1].strip().lower()

            # Remove common filler words
            for filler in ["um", "uh", "like", "you know", "so"]:
                if last_sentence.endswith(filler):
                    last_sentence = last_sentence[: -len(filler)].strip()

            # Check for proper sentence ending
            return last_sentence.endswith((".", "!", "?"))

        segments = transcription["segments"]
        chunks = []
        current_chunk = {
            "start": 0,
            "end": 0,
            "text": [],
            "segments": [],
            "confidence": [],
        }

        for i, segment in enumerate(segments):
            segment_start = int(segment["start"] * 1000)
            segment_end = int(segment["end"] * 1000)
            segment_text = segment["text"].strip()

            # Calculate current chunk duration
            chunk_duration = segment_end - current_chunk["start"]

            # Get semantic similarity with current chunk
            current_text = " ".join(current_chunk["text"])
            semantic_similarity = (
                calculate_semantic_similarity(current_text, segment_text)
                if current_text
                else 1.0
            )

            # Determine if we should create a new chunk
            should_break = chunk_duration >= self.max_chunk_duration or (
                chunk_duration >= self.min_chunk_duration
                and is_complete_thought(segment_text)
                and semantic_similarity < 0.3
            )

            if should_break:
                # Finalize current chunk
                if current_chunk["text"]:
                    current_chunk["text"] = " ".join(current_chunk["text"])
                    chunks.append(current_chunk)

                # Start new chunk
                current_chunk = {
                    "start": segment_end,
                    "end": segment_end,
                    "text": [segment_text],
                    "segments": [segment],
                    "confidence": [segment.get("confidence", 1.0)],
                }
            else:
                # Add to current chunk
                current_chunk["text"].append(segment_text)
                current_chunk["segments"].append(segment)
                current_chunk["confidence"].append(segment.get("confidence", 1.0))
                current_chunk["end"] = segment_end

        # Add the last chunk if it contains content
        if current_chunk["text"]:
            current_chunk["text"] = " ".join(current_chunk["text"])
            chunks.append(current_chunk)

        # Post-process chunks
        return self.post_process_chunks(chunks)

    def post_process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Optimize chunk boundaries and merge short segments"""
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Refine chunk boundaries
            chunk = self.refine_chunk_boundaries(chunk)

            # Calculate chunk metrics
            chunk_duration = chunk["end"] - chunk["start"]

            # Merge short chunks with previous if appropriate
            if i > 0 and chunk_duration < self.min_chunk_duration:
                prev_chunk = processed_chunks[-1]
                combined_duration = (
                    prev_chunk["end"] - prev_chunk["start"] + chunk_duration
                )

                if combined_duration <= self.max_chunk_duration:
                    # Merge with previous chunk
                    processed_chunks[-1] = self.merge_chunks(prev_chunk, chunk)
                    continue

            processed_chunks.append(chunk)

        return processed_chunks

    def refine_chunk_boundaries(self, chunk: Dict) -> Dict:
        """Refine chunk boundaries based on voice activity"""
        try:
            # Load audio segment
            audio_segment = self._load_audio_segment(chunk["start"], chunk["end"])

            # Detect voice activity
            voiced_frames = self.detect_voice_activity(audio_segment)

            if voiced_frames:
                # Adjust boundaries to voice activity
                chunk["start"] = chunk["start"] + voiced_frames[0].timestamp * 1000
                chunk["end"] = chunk["start"] + voiced_frames[-1].timestamp * 1000
        except Exception as e:
            logger.warning(f"Could not refine chunk boundaries: {str(e)}")

        return chunk

    def merge_chunks(self, chunk1: Dict, chunk2: Dict) -> Dict:
        """Merge two chunks while preserving metadata"""
        return {
            "start": chunk1["start"],
            "end": chunk2["end"],
            "text": chunk1["text"] + " " + chunk2["text"],
            "segments": chunk1["segments"] + chunk2["segments"],
            "confidence": chunk1["confidence"] + chunk2["confidence"],
        }

    def validate_chunk_quality(self, chunk: Dict) -> float:
        """Validate the quality of a chunk"""
        if not self.validate_chunks:
            return 1.0

        scores = {
            "duration": self._score_duration(chunk),
            "confidence": np.mean(chunk["confidence"]),
            "semantic_completeness": self._score_semantic_completeness(chunk),
            "audio_quality": self._score_audio_quality(chunk),
        }

        return np.mean(list(scores.values()))

    def _score_duration(self, chunk: Dict) -> float:
        """Score chunk duration"""
        duration = chunk["end"] - chunk["start"]
        if duration < self.min_chunk_duration:
            return duration / self.min_chunk_duration
        if duration > self.max_chunk_duration:
            return 1 - (duration - self.max_chunk_duration) / self.max_chunk_duration
        return 1.0

    def _score_semantic_completeness(self, chunk: Dict) -> float:
        """Score semantic completeness of chunk"""
        text = chunk["text"]
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return 0.0

        scores = []
        for sentence in sentences:
            # Check for complete sentence structure
            has_ending = sentence.strip().endswith((".", "!", "?"))
            words = len(sentence.split())
            reasonable_length = 3 <= words <= 50

            score = (has_ending + reasonable_length) / 2
            scores.append(score)

        return np.mean(scores)

    def _score_audio_quality(self, chunk: Dict) -> float:
        """Score audio quality of chunk"""
        try:
            audio_segment = self._load_audio_segment(chunk["start"], chunk["end"])
            voiced_frames = self.detect_voice_activity(audio_segment)
            return len(voiced_frames) / (
                len(audio_segment) / 480
            )  # 480 samples per 30ms frame
        except:
            return 0.8  # Default score if audio analysis fails

    def process_chunk(self, chunk_data: Dict) -> Dict:
        """Process a single audio chunk with quality validation"""
        try:
            chunk_id = chunk_data["chunk_id"]
            input_path = chunk_data["input_path"]
            start_time = chunk_data["start"] / 1000  # Convert to seconds
            end_time = chunk_data["end"] / 1000

            output_filename = f"chunk_{chunk_id:04d}.wav"
            output_path = str(self.output_dir / "chunks" / output_filename)

            # Process the audio chunk
            self.process_audio_chunk(input_path, output_path, start_time, end_time)

            # Validate chunk quality
            quality_score = self.validate_chunk_quality(chunk_data)

            return {
                "chunk_id": chunk_id,
                "audio_file": output_filename,
                "start_time": start_time,
                "end_time": end_time,
                "text": chunk_data["text"],
                "segments": chunk_data["segments"],
                "quality_score": quality_score,
                "confidence": np.mean(chunk_data["confidence"]),
            }
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            raise

    def save_chunks(self, audio_path: str, chunks: List[Dict]):
        """Save audio chunks and their transcriptions using parallel processing"""
        logger.info("Saving chunks with parallel processing")
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Prepare chunk data for parallel processing
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append(
                {
                    "chunk_id": i,
                    "input_path": audio_path,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "segments": chunk["segments"],
                    "confidence": chunk.get("confidence", [1.0]),
                }
            )

        # Process chunks in parallel with optimal worker count
        metadata = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk): chunk
                for chunk in chunk_data
            }

            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    try:
                        result = future.result()
                        metadata.append(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {str(e)}")

        # Sort metadata by chunk_id to maintain order
        metadata.sort(key=lambda x: x["chunk_id"])

        # Save metadata with quality metrics
        try:
            with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    def process_audio_chunk(
        self, input_path: str, output_path: str, start_time: float, end_time: float
    ):
        """Extract and save an audio chunk from the input file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Use ffmpeg to extract the chunk
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-i",
                    input_path,
                    "-ss",
                    str(start_time),
                    "-to",
                    str(end_time),
                    "-acodec",
                    "pcm_s16le",  # Use same codec as input
                    "-ar",
                    "16000",  # Set sample rate
                    "-ac",
                    "1",  # Convert to mono
                    output_path,
                ],
                check=True,
                capture_output=True,
            )

            logger.info(f"Saved audio chunk to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio chunk: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            raise

    def _load_audio_segment(self, start_ms: int, end_ms: int) -> np.ndarray:
        """Load an audio segment from the specified time range"""
        try:
            # Convert milliseconds to seconds
            start_time = start_ms / 1000
            end_time = end_ms / 1000

            # Create a temporary file for the segment
            temp_file = str(self.output_dir / f"temp_segment_{start_ms}_{end_ms}.wav")

            # Extract segment using ffmpeg
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.current_audio_path,
                    "-ss",
                    str(start_time),
                    "-to",
                    str(end_time),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    temp_file,
                ],
                check=True,
                capture_output=True,
            )

            # Read the audio data
            audio_data = np.fromfile(temp_file, dtype=np.int16)

            # Clean up temporary file
            os.remove(temp_file)

            return audio_data
        except Exception as e:
            logger.error(f"Error loading audio segment: {str(e)}")
            raise

    def process_video(self, youtube_url: str) -> Dict:
        """Process entire video from URL to semantic chunks with quality metrics"""
        try:
            # Download and preprocess audio
            audio_path = self.download_video(youtube_url)
            self.current_audio_path = audio_path  # Store the current audio path
            logger.info(f"Audio saved to: {audio_path}")

            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            logger.info("Transcription completed")

            # Create semantic chunks
            chunks = self.create_semantic_chunks(audio_path, transcription)
            logger.info(f"Created {len(chunks)} semantic chunks")

            # Save chunks and metadata
            self.save_chunks(
                self.current_audio_path, chunks
            )  # Use the current audio path
            logger.info("All chunks saved successfully")

            # Calculate metrics
            total_duration = sum(
                (chunk["end"] - chunk["start"]) / 1000 for chunk in chunks
            )
            avg_confidence = np.mean(
                [np.mean(chunk.get("confidence", [1.0])) for chunk in chunks]
            )

            metrics = {
                "audio_path": self.current_audio_path,
                "num_chunks": len(chunks),
                "total_duration": total_duration,
                "avg_confidence": avg_confidence,
                "output_dir": str(self.output_dir),
            }

            # Log metrics
            logger.info("Metrics", extra={"metrics": metrics})

            return metrics
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise


class Frame:
    """A frame of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


if __name__ == "__main__":
    try:
        # Example usage
        youtube_url = "https://www.youtube.com/watch?v=-uleG_Vecis"
        segmenter = YoutubeSemanticSegmenter(
            preprocess_audio=True, validate_chunks=True
        )
        result = segmenter.process_video(youtube_url)

        logger.info("\nProcessing complete!")
        logger.info(f"Audio saved to: {result['audio_path']}")
        logger.info(f"Created {result['num_chunks']} semantic chunks")
        logger.info(f"Average confidence: {result['avg_confidence']:.2f}")
        logger.info(f"Output directory: {result['output_dir']}")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
