# Referencing workspace paths
import sys
sys.path.insert(0, r"c:/Users/Om Thakare/OneDrive/Desktop/DataThon_PS6")

import streamlit as st
import os
import json
from youtube_semantic_segmenter import YoutubeSemanticSegmenter  # [youtube_semantic_segmenter.py](youtube_semantic_segmenter.py)

def main():
    st.set_page_config(page_title="Advanced Semantic Segmenter", layout="wide")
    st.title("YouTube Semantic Segmenter - Advanced")

    with st.sidebar:
        st.header("Configuration")
        model_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large", "large-v3"], index=5)
        preprocess_audio = st.checkbox("Preprocess Audio", value=True)
        validate_chunks = st.checkbox("Validate Chunks", value=True)

    url = st.text_input("Enter YouTube URL")
    if st.button("Process Video"):
        segmenter = YoutubeSemanticSegmenter(
            model_size=model_size,
            preprocess_audio=preprocess_audio,
            validate_chunks=validate_chunks,
        )
        with st.spinner("Processing..."):
            result = segmenter.process_video(url)

        st.success("Processing complete!")
        st.write(f"Audio Path: {result['audio_path']}")
        st.write(f"Chunks Created: {result['num_chunks']}")
        st.write(f"Average Confidence: {result['avg_confidence']:.2f}")

        metadata_file = os.path.join(result["C:\Users\Om Thakare\OneDrive\Desktop\DataThon_PS6\output"], "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            st.write("Metadata Loaded:")
            for chunk in metadata:
                st.markdown(f"**Chunk {chunk['chunk_id']}** - Score: {chunk['quality_score']:.2f}")
                st.write(chunk["text"])

if __name__ == "__main__":
    main()