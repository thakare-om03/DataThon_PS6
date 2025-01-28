#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from youtube_semantic_segmenter import YoutubeSemanticSegmenter
import json


def setup_logging(verbose: bool):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("semantic_segmenter.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract semantically meaningful segments from YouTube videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("url", help="YouTube video URL to process")

    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Directory to store output files",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-detected based on system resources)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--max-chunk-duration",
        type=float,
        default=40.0,
        help="Maximum duration of each chunk in seconds",
    )

    parser.add_argument(
        "--min-chunk-duration",
        type=float,
        default=20.0,
        help="Minimum duration of each chunk in seconds",
    )

    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        default="large-v3",
        help="Whisper model size to use",
    )

    parser.add_argument(
        "--preprocess-audio",
        action="store_true",
        help="Apply audio preprocessing for better quality",
    )

    parser.add_argument(
        "--no-preprocess-audio",
        action="store_false",
        dest="preprocess_audio",
        help="Disable audio preprocessing",
    )

    parser.add_argument(
        "--validate-chunks",
        action="store_true",
        help="Enable chunk quality validation",
    )

    parser.add_argument(
        "--no-validate-chunks",
        action="store_false",
        dest="validate_chunks",
        help="Disable chunk quality validation",
    )

    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum quality score for chunks (0.0-1.0)",
    )

    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help="Collect and save detailed metrics",
    )

    # Set defaults for new boolean flags
    parser.set_defaults(
        preprocess_audio=True, validate_chunks=True, collect_metrics=True
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Initialize segmenter with new parameters
        segmenter = YoutubeSemanticSegmenter(
            output_dir=args.output_dir,
            num_workers=args.workers,
            max_chunk_duration=args.max_chunk_duration,
            min_chunk_duration=args.min_chunk_duration,
            model_size=args.model_size,
            preprocess_audio=args.preprocess_audio,
            validate_chunks=args.validate_chunks,
        )

        # Process video
        logger.info(f"Processing video: {args.url}")
        result = segmenter.process_video(args.url)

        # Print results with additional metrics
        logger.info(f"\nProcessing complete!")
        logger.info(f"Audio saved to: {result['audio_path']}")
        logger.info(f"Generated {result['num_chunks']} chunks")
        logger.info(f"Total duration processed: {result['total_duration']:.2f} seconds")
        logger.info(f"Average confidence score: {result['avg_confidence']:.2f}")

        # Save detailed metrics if requested
        if args.collect_metrics:
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(segmenter.metrics_handler.metrics, f, indent=2)
            logger.info(f"Detailed metrics saved to: {metrics_path}")

        # Quality validation summary
        if args.validate_chunks:
            low_quality_chunks = [
                m
                for m in segmenter.metrics_handler.metrics
                if m.get("quality_score", 1.0) < args.min_quality_score
            ]
            if low_quality_chunks:
                logger.warning(
                    f"Found {len(low_quality_chunks)} chunks below quality threshold "
                    f"({args.min_quality_score})"
                )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
