import streamlit as st

# Configure page
st.set_page_config(
    page_title="Help - YouTube Semantic Segmenter", page_icon="❓", layout="wide"
)

# Add custom CSS
st.markdown(
    """
    <style>
    .help-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #ddd;
    }
    .faq-question {
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 10px;
    }
    .faq-answer {
        margin-bottom: 20px;
        padding-left: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def main():
    st.title("❓ Help & Documentation")
    st.markdown(
        """
    Welcome to the YouTube Semantic Segmenter help page. Here you'll find detailed information about using the application,
    troubleshooting common issues, and answers to frequently asked questions.
    """
    )

    # Quick Start Guide
    st.header("🚀 Quick Start Guide")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown(
        """
    1. **Enter YouTube URL**: Paste the URL of the YouTube video you want to process
    2. **Configure Settings**: Adjust the processing settings in the sidebar (optional)
    3. **Process Video**: Click the "Process Video" button to start
    4. **View Results**: Once processing is complete, you can:
        - Download individual chunks
        - View transcripts
        - Analyze quality metrics
        - Export results
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Features
    st.header("✨ Features")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown(
        """
    - **Semantic Segmentation**: Intelligently splits videos into meaningful chunks
    - **Quality Validation**: Ensures high-quality transcription and segmentation
    - **Multiple Export Options**: Download audio chunks and transcripts
    - **Analytics Dashboard**: Visualize processing metrics and results
    - **Configurable Settings**: Customize processing parameters
    - **Batch Processing**: Process multiple videos efficiently
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Configuration Guide
    st.header("⚙️ Configuration Guide")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Model Settings
    - **Model Size**: Choose between different Whisper model sizes
        - `tiny`: Fastest but less accurate
        - `base`: Good balance of speed and accuracy
        - `small`: Better accuracy, slower processing
        - `medium`: High accuracy, slower processing
        - `large-v3`: Best accuracy, slowest processing

    ### Audio Processing
    - **Sample Rate**: Higher rates capture more audio detail
    - **Audio Format**: Choose between WAV, MP3, or FLAC
    - **Normalization**: Improves audio quality for better transcription
    - **Silence Removal**: Removes silent segments from audio

    ### Chunking Settings
    - **Min/Max Duration**: Control chunk length
    - **Overlap Duration**: Add overlap between chunks
    - **Quality Threshold**: Set minimum quality requirements
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Troubleshooting
    st.header("🔧 Troubleshooting")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Common Issues

    1. **Video Download Fails**
        - Check if the video is available in your region
        - Verify the URL is correct
        - Ensure you have internet connectivity

    2. **Processing Takes Too Long**
        - Try using a smaller model size
        - Reduce video length
        - Check system resources

    3. **Poor Quality Results**
        - Use a larger model size
        - Enable audio preprocessing
        - Adjust quality thresholds
        - Check input audio quality

    4. **Application Crashes**
        - Clear browser cache
        - Restart the application
        - Check system memory
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ
    st.header("❓ Frequently Asked Questions")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)

    # FAQ 1
    st.markdown('<div class="faq-question">', unsafe_allow_html=True)
    st.markdown("Q: What video formats are supported?")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">', unsafe_allow_html=True)
    st.markdown(
        """
    The application supports any video format that YouTube-DL can handle, which includes:
    - MP4
    - WebM
    - FLV
    - And most other common formats
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ 2
    st.markdown('<div class="faq-question">', unsafe_allow_html=True)
    st.markdown("Q: How long does processing take?")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">', unsafe_allow_html=True)
    st.markdown(
        """
    Processing time depends on:
    - Video length
    - Model size
    - System resources
    - Network speed
    
    A 10-minute video typically takes 2-5 minutes with the base model.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ 3
    st.markdown('<div class="faq-question">', unsafe_allow_html=True)
    st.markdown("Q: Can I process videos in other languages?")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">', unsafe_allow_html=True)
    st.markdown(
        """
    Yes! The Whisper model supports multiple languages. The application will automatically:
    - Detect the spoken language
    - Transcribe accordingly
    - Provide translations if needed
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ 4
    st.markdown('<div class="faq-question">', unsafe_allow_html=True)
    st.markdown("Q: How can I improve transcription quality?")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">', unsafe_allow_html=True)
    st.markdown(
        """
    To improve transcription quality:
    1. Use a larger model size (e.g., large-v3)
    2. Enable audio preprocessing
    3. Ensure good quality input audio
    4. Adjust chunk durations appropriately
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Support
    st.header("📧 Support")
    st.markdown('<div class="help-section">', unsafe_allow_html=True)
    st.markdown(
        """
    Need additional help? You can:
    
    - Visit our [GitHub repository](https://github.com/achalbajpai/youtube-semantic-segmenter)
    - Open an [issue](https://github.com/achalbajpai/youtube-semantic-segmenter/issues)
    - Check the [documentation](https://github.com/achalbajpai/youtube-semantic-segmenter/wiki)
    - Contact the maintainers
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
