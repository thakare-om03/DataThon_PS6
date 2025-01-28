#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def print_chunk(chunk: Dict, verbose: bool = False):
    """Print chunk information in a formatted way"""
    chunk_id = chunk["chunk_id"]
    start = format_time(chunk["start_time"])
    end = format_time(chunk["end_time"])

    print(f"\n{'='*80}")
    print(f"Chunk {chunk_id:02d} [{start} - {end}]")
    print(f"{'-'*80}")
    print(f"Text: {chunk['text']}")

    if verbose:
        print(f"{'-'*80}")
        print(f"Quality Score: {chunk['quality_score']:.2f}")
        print(f"Confidence: {chunk['confidence']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="View transcripts from metadata.json")
    parser.add_argument(
        "--metadata",
        type=str,
        default="output/metadata.json",
        help="Path to metadata.json file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show additional information like quality scores",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        help="Show specific chunk number only",
    )
    args = parser.parse_args()

    # Check if metadata file exists
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: {args.metadata} not found!")
        print("Please run the segmenter first to generate metadata.")
        sys.exit(1)

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {args.metadata} is not a valid JSON file!")
        sys.exit(1)

    if args.chunk is not None:
        # Show specific chunk
        for chunk in metadata:
            if chunk["chunk_id"] == args.chunk:
                print_chunk(chunk, args.verbose)
                break
        else:
            print(f"Error: Chunk {args.chunk} not found!")
    else:
        # Show all chunks
        for chunk in metadata:
            print_chunk(chunk, args.verbose)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Total chunks: {len(metadata)}")


if __name__ == "__main__":
    main()
