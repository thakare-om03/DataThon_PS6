import streamlit as st
import os
import json
from pathlib import Path
from youtube_semantic_segmenter import YoutubeSemanticSegmenter
import plotly.graph_objects as go
import pandas as pd
import time

# Configure page
st.set_page_config(
    page_title="YouTube Semantic Segmenter",
    page_icon="🎬",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/achalbajpai/youtube-semantic-segmenter",
        "Report a bug": "https://github.com/achalbajpai/youtube-semantic-segmenter/issues",
        "About": "A tool that segments YouTube videos into semantically meaningful chunks with transcripts.",
    },
)

# Add custom CSS
st.markdown(
    """
    <style>
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    .chunk-box {
        border: 1px solid #ddd;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def display_chunk_metrics(metadata):
    """Display metrics using Plotly"""
    df = pd.DataFrame(metadata)

    # Duration distribution
    durations = [end - start for start, end in zip(df["start_time"], df["end_time"])]
    fig_duration = go.Figure(data=[go.Histogram(x=durations, nbinsx=20)])
    fig_duration.update_layout(
        title="Chunk Duration Distribution",
        xaxis_title="Duration (seconds)",
        yaxis_title="Count",
        template="plotly_white",
    )
    st.plotly_chart(fig_duration, use_container_width=True)

    # Quality scores
    fig_quality = go.Figure()
    fig_quality.add_trace(go.Box(y=df["quality_score"], name="Quality Scores"))
    fig_quality.add_trace(go.Box(y=df["confidence"], name="Confidence Scores"))
    fig_quality.update_layout(
        title="Quality and Confidence Distribution",
        yaxis_title="Score",
        template="plotly_white",
    )
    st.plotly_chart(fig_quality, use_container_width=True)


def main():
    # Header
    st.title("🎬 YouTube Semantic Segmenter")
    st.markdown(
        """
    Transform YouTube videos into semantically meaningful chunks with transcripts. 
    This tool uses OpenAI's Whisper model for transcription and advanced algorithms for semantic segmentation.
    """
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Model Settings")
        model_size = st.selectbox(
            "Model Size",
            ["base", "large-v3"],
            index=1,
            help="Choose between faster (base) or more accurate (large-v3) model",
        )

        st.subheader("Chunk Settings")
        max_duration = st.slider(
            "Max Chunk Duration (seconds)",
            5,
            30,
            15,
            help="Maximum duration of each chunk",
        )

        min_duration = st.slider(
            "Min Chunk Duration (seconds)",
            3,
            15,
            5,
            help="Minimum duration of each chunk",
        )

        st.subheader("Processing Options")
        preprocess_audio = st.checkbox(
            "Preprocess Audio",
            True,
            help="Apply audio preprocessing for better quality",
        )

        validate_chunks = st.checkbox(
            "Validate Chunks", True, help="Enable quality validation for chunks"
        )

        st.markdown("---")
        st.markdown(
            """
        ### About
        This tool helps you:
        - Download YouTube videos
        - Transcribe audio content
        - Create semantic chunks
        - Analyze quality metrics
        """
        )

    # Main interface
    st.header("🎯 Process Video")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter the full YouTube video URL",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        process_button = st.button(
            "🚀 Process Video", type="primary", use_container_width=True
        )
    with col2:
        output_format = st.selectbox(
            "Output Format", ["WAV", "MP3"], help="Choose the output audio format"
        )
    with col3:
        download_transcripts = st.checkbox(
            "Download Transcripts", True, help="Generate downloadable transcript files"
        )

    if process_button:
        if not youtube_url:
            st.error("⚠️ Please enter a YouTube URL")
            return

        try:
            # Create processing container
            process_container = st.container()
            with process_container:
                st.markdown("### 📊 Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize segmenter
                segmenter = YoutubeSemanticSegmenter(
                    output_dir="output",
                    max_chunk_duration=float(max_duration),
                    min_chunk_duration=float(min_duration),
                    model_size=model_size,
                    preprocess_audio=preprocess_audio,
                    validate_chunks=validate_chunks,
                )

                # Download and process video
                status_text.text("⬇️ Downloading video...")
                progress_bar.progress(20)

                result = segmenter.process_video(youtube_url)
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Success message
                st.success(
                    f"🎉 Successfully processed video into {result['num_chunks']} chunks!"
                )

                # Load and display metadata
                with open(
                    os.path.join(result["output_dir"], "metadata.json"), "r"
                ) as f:
                    metadata = json.load(f)

                # Display overall metrics
                st.markdown("### 📈 Overall Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Chunks", result["num_chunks"])
                with col2:
                    st.metric("Total Duration", f"{result['total_duration']:.2f}s")
                with col3:
                    st.metric("Avg. Confidence", f"{result['avg_confidence']:.2%}")
                with col4:
                    st.metric("Processing Time", f"{time.time():.1f}s")

                # Display analytics
                st.markdown("### 📊 Analytics")
                display_chunk_metrics(metadata)

                # Display chunks
                st.markdown("### 📝 Chunks")
                for chunk in metadata:
                    with st.expander(
                        f"Chunk {chunk['chunk_id']:02d} [{format_time(chunk['start_time'])} - {format_time(chunk['end_time'])}]"
                    ):
                        st.markdown(f"**Text:** {chunk['text']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quality Score", f"{chunk['quality_score']:.2f}")
                        with col2:
                            st.metric("Confidence", f"{chunk['confidence']:.2f}")
                        with col3:
                            st.metric(
                                "Duration",
                                f"{chunk['end_time'] - chunk['start_time']:.1f}s",
                            )

                        # Download buttons
                        download_col1, download_col2 = st.columns(2)
                        with download_col1:
                            st.download_button(
                                "⬇️ Download Audio",
                                data=open(
                                    f"output/chunks/chunk_{chunk['chunk_id']:04d}.wav",
                                    "rb",
                                ),
                                file_name=f"chunk_{chunk['chunk_id']:04d}.wav",
                                mime="audio/wav",
                            )
                        with download_col2:
                            st.download_button(
                                "📄 Download Transcript",
                                data=chunk["text"],
                                file_name=f"chunk_{chunk['chunk_id']:04d}.txt",
                                mime="text/plain",
                            )

        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
