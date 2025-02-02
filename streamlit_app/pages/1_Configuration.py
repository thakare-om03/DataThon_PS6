import streamlit as st
import json
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Configuration - YouTube Semantic Segmenter",
    page_icon="⚙️",
    layout="wide",
)

# Add custom CSS
st.markdown(
    """
    <style>
    .config-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def save_config(config):
    """Save configuration to JSON file"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    with open(config_dir / "app_config.json", "w") as f:
        json.dump(config, f, indent=4)

    st.success("✅ Configuration saved successfully!")


def load_config():
    """Load configuration from JSON file"""
    config_path = Path("config/app_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def main():
    st.title("⚙️ Configuration")
    st.markdown(
        """
    Configure the default settings for the YouTube Semantic Segmenter. 
    These settings will be used as defaults when processing videos.
    """
    )

    # Load existing configuration
    config = load_config()

    # Model Configuration
    st.header("🤖 Model Configuration")
    with st.container():
        st.markdown('<div class="config-box">', unsafe_allow_html=True)

        model_size = st.selectbox(
            "Default Model Size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=["tiny", "base", "small", "medium", "large-v3"].index(
                config.get("model_size", "large-v3")
            ),
            help="Choose the default Whisper model size",
        )

        device = st.selectbox(
            "Processing Device",
            ["cuda", "cpu"],
            index=["cuda", "cpu"].index(config.get("device", "cuda")),
            help="Choose the device for model inference",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Audio Processing Configuration
    st.header("🎵 Audio Processing")
    with st.container():
        st.markdown('<div class="config-box">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.number_input(
                "Sample Rate (Hz)",
                min_value=8000,
                max_value=48000,
                value=config.get("sample_rate", 16000),
                step=1000,
                help="Audio sample rate for processing",
            )

        with col2:
            audio_format = st.selectbox(
                "Default Audio Format",
                ["WAV", "MP3", "FLAC"],
                index=["WAV", "MP3", "FLAC"].index(config.get("audio_format", "WAV")),
                help="Choose the default audio format for chunks",
            )

        normalize_audio = st.checkbox(
            "Normalize Audio",
            value=config.get("normalize_audio", True),
            help="Apply audio normalization during preprocessing",
        )

        remove_silence = st.checkbox(
            "Remove Silence",
            value=config.get("remove_silence", True),
            help="Remove silent segments from audio",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Chunking Configuration
    st.header("✂️ Chunking Settings")
    with st.container():
        st.markdown('<div class="config-box">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            min_chunk_duration = st.slider(
                "Default Minimum Chunk Duration (seconds)",
                min_value=1,
                max_value=30,
                value=config.get("min_chunk_duration", 5),
                help="Minimum duration for each chunk",
            )

        with col2:
            max_chunk_duration = st.slider(
                "Default Maximum Chunk Duration (seconds)",
                min_value=5,
                max_value=60,
                value=config.get("max_chunk_duration", 15),
                help="Maximum duration for each chunk",
            )

        overlap_duration = st.slider(
            "Chunk Overlap Duration (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=float(config.get("overlap_duration", 0.5)),
            step=0.1,
            help="Duration of overlap between consecutive chunks",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Output Configuration
    st.header("📁 Output Settings")
    with st.container():
        st.markdown('<div class="config-box">', unsafe_allow_html=True)

        output_dir = st.text_input(
            "Output Directory",
            value=config.get("output_dir", "output"),
            help="Directory to store processed files",
        )

        save_transcripts = st.checkbox(
            "Save Transcripts",
            value=config.get("save_transcripts", True),
            help="Generate and save transcript files",
        )

        save_metadata = st.checkbox(
            "Save Metadata",
            value=config.get("save_metadata", True),
            help="Save processing metadata and analytics",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Save Configuration
    if st.button("💾 Save Configuration", type="primary"):
        new_config = {
            "model_size": model_size,
            "device": device,
            "sample_rate": sample_rate,
            "audio_format": audio_format,
            "normalize_audio": normalize_audio,
            "remove_silence": remove_silence,
            "min_chunk_duration": min_chunk_duration,
            "max_chunk_duration": max_chunk_duration,
            "overlap_duration": overlap_duration,
            "output_dir": output_dir,
            "save_transcripts": save_transcripts,
            "save_metadata": save_metadata,
        }
        save_config(new_config)

    # Reset Configuration
    if st.button("🔄 Reset to Defaults", type="secondary"):
        if os.path.exists("config/app_config.json"):
            os.remove("config/app_config.json")
            st.success("✅ Configuration reset to defaults!")
            st.experimental_rerun()


if __name__ == "__main__":
    main()
