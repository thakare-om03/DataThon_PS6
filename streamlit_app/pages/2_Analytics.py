import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Analytics - YouTube Semantic Segmenter", page_icon="📊", layout="wide"
)

# Add custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
        text-align: center;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #eee;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def load_metadata():
    """Load all metadata files from the output directory"""
    metadata_files = list(Path("output").glob("**/metadata.json"))
    all_metadata = []

    for file in metadata_files:
        try:
            with open(file, "r") as f:
                metadata = json.load(f)
                # Add file creation time
                metadata["processed_at"] = datetime.fromtimestamp(file.stat().st_ctime)
                all_metadata.extend(
                    metadata if isinstance(metadata, list) else [metadata]
                )
        except Exception as e:
            st.warning(f"Error loading metadata from {file}: {str(e)}")

    return pd.DataFrame(all_metadata)


def plot_duration_distribution(df):
    """Plot chunk duration distribution"""
    durations = df.apply(lambda x: x["end_time"] - x["start_time"], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=durations, nbinsx=30, name="Duration Distribution"))

    fig.update_layout(
        title="Chunk Duration Distribution",
        xaxis_title="Duration (seconds)",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_quality_metrics(df):
    """Plot quality metrics distribution"""
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=df["quality_score"],
            name="Quality Score",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        )
    )

    fig.add_trace(
        go.Box(
            y=df["confidence"],
            name="Confidence Score",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        )
    )

    fig.update_layout(
        title="Quality Metrics Distribution",
        yaxis_title="Score",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_processing_timeline(df):
    """Plot processing timeline"""
    df_timeline = df.copy()
    df_timeline["date"] = pd.to_datetime(df_timeline["processed_at"]).dt.date
    daily_counts = df_timeline.groupby("date").size().reset_index(name="count")

    fig = px.line(
        daily_counts,
        x="date",
        y="count",
        title="Processing Timeline",
        template="plotly_white",
    )

    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Chunks Processed")

    return fig


def plot_text_length_distribution(df):
    """Plot text length distribution"""
    df["text_length"] = df["text"].str.len()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=df["text_length"], nbinsx=30, name="Text Length Distribution")
    )

    fig.update_layout(
        title="Text Length Distribution",
        xaxis_title="Number of Characters",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def main():
    st.title("📊 Analytics Dashboard")
    st.markdown(
        """
    Analyze the performance and results of your video processing tasks.
    This dashboard provides insights into chunk quality, duration distribution, and processing metrics.
    """
    )

    # Load data
    try:
        df = load_metadata()
        if df.empty:
            st.warning(
                "No processed videos found. Process some videos to see analytics."
            )
            return
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return

    # Overall Metrics
    st.header("📈 Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Chunks", len(df), f"From {df['processed_at'].nunique()} videos"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_duration = df.apply(
            lambda x: x["end_time"] - x["start_time"], axis=1
        ).mean()
        st.metric(
            "Average Duration",
            f"{avg_duration:.2f}s",
            f"±{df.apply(lambda x: x['end_time'] - x['start_time'], axis=1).std():.2f}s",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Average Quality",
            f"{df['quality_score'].mean():.2%}",
            f"{df['quality_score'].std():.2%}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Average Confidence",
            f"{df['confidence'].mean():.2%}",
            f"{df['confidence'].std():.2%}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    st.header("📊 Detailed Analysis")

    # Duration Distribution
    st.subheader("Duration Analysis")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_duration_distribution(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quality Metrics
    st.subheader("Quality Metrics")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_quality_metrics(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Processing Timeline
    st.subheader("Processing Timeline")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_processing_timeline(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Text Length Distribution
    st.subheader("Text Analysis")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_text_length_distribution(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Data Table
    st.header("🔍 Detailed Data")
    st.dataframe(
        df[
            [
                "chunk_id",
                "start_time",
                "end_time",
                "quality_score",
                "confidence",
                "text",
            ]
        ].sort_values("chunk_id"),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
