from scipy import signal
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from feat import Detector
from feat.utils.io import read_feat
import cv2
from PIL import Image
import tempfile
import base64
from io import BytesIO

# Initialize the detector
detector = Detector()

# Dictionary to map AU codes to their names
au_names = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU28": "Lip Suck",
    "AU43": "Eyes Closed",
}

# Define AUs for happiness (Goodnews) and sadness (Badnews)
happiness_aus = ["AU06", "AU12", "AU25", "AU26"]  # Goodnews AUs
sadness_aus = ["AU01", "AU04", "AU15", "AU17", "AU24"]  # Badnews AUs

# Function to create a download link for a DataFrame
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to create a download link for a plot
def get_plot_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generates a link allowing the plot to be downloaded"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to process a single video
def process_single_video(input_video_path, output_video_path, confidence_threshold, selected_aus):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return pd.DataFrame(), {}

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width + 400, height))

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=["Frame", "Emotion", "AU", "Confidence"])
    
    # Initialize summary statistics
    summary_stats = {
        "Total Frames": total_frames,
        "Frames Processed": 0,
        "Frames with Faces": 0,
        "Average Confidence": 0,
        "Dominant Emotion": None,
        "Most Active AU": None,
        "Average Goodnews Intensity": 0,
        "Average Badnews Intensity": 0
    }
    
    emotion_counts = {}
    au_intensities = {}

    # Placeholders for displays
    video_placeholder = st.empty()
    cropped_face_placeholder = st.empty()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL image and save temporarily
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_image_path = temp_file.name
            pil_image.save(temp_image_path, format="JPEG")

        # Detect facial expressions
        results = detector.detect_image([temp_image_path])
        os.remove(temp_image_path)

        if not results.empty:
            face = results.iloc[0]
            summary_stats["Frames with Faces"] += 1

            # Draw landmarks if available
            if "landmarks" in results.columns:
                landmarks = face["landmarks"]
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Draw face bounding box if available
            if all(col in results.columns for col in ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]):
                x, y, w, h = face["FaceRectX"], face["FaceRectY"], face["FaceRectWidth"], face["FaceRectHeight"]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                
                # Display cropped face
                cropped_face = frame[int(y):int(y + h), int(x):int(x + w)]
                cropped_face_placeholder.image(cropped_face, channels="BGR", use_container_width=True)

            # Display confidence score
            if "confidence" in results.columns:
                confidence = face["confidence"]
                summary_stats["Average Confidence"] += confidence
                confidence_text = f"Confidence: {confidence:.2f}"
                text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = width - text_size[0] - 10
                text_y = text_size[1] + 10
                text_color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)
                cv2.putText(frame, confidence_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Create side panel
            side_panel = np.zeros((height, 400, 3), dtype=np.uint8)
            side_panel.fill(255)

            # Plot emotions
            emotion_columns = ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "neutral"]
            emotion_values = [face[col] for col in emotion_columns if col in results.columns]
            
            # Update emotion counts
            dominant_emotion = emotion_columns[np.argmax(emotion_values)]
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
            
            plt.figure(figsize=(4, 4))
            plt.barh(emotion_columns, emotion_values, color=['green', 'blue', 'red', 'purple', 'orange', 'brown', 'gray'])
            plt.title("Emotions")
            plt.xlim(0, 1)
            plt.tight_layout()
            temp_chart_path = "temp_chart.png"
            plt.savefig(temp_chart_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            chart_image = cv2.imread(temp_chart_path)
            chart_height, chart_width, _ = chart_image.shape
            side_panel[10:10 + chart_height, 10:10 + chart_width] = chart_image

            # Plot selected AUs
            au_columns = [col for col in results.columns if col.startswith("AU") and col in selected_aus]
            au_values = [face[col] for col in au_columns]
            au_labels = [f"{col} - {au_names.get(col, 'Unknown')}" for col in au_columns]

            # Update AU intensities and Goodnews/Badnews stats
            for au, value in zip(au_columns, au_values):
                au_intensities[au] = au_intensities.get(au, 0) + value
                if au in happiness_aus:
                    summary_stats["Average Goodnews Intensity"] += value
                elif au in sadness_aus:
                    summary_stats["Average Badnews Intensity"] += value
            
            plt.figure(figsize=(4, 4))
            plt.barh(au_labels, au_values, color='cyan')
            plt.title("Action Units")
            plt.xlim(0, 1)
            plt.tight_layout()
            plt.savefig(temp_chart_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            chart_image = cv2.imread(temp_chart_path)
            chart_height, chart_width, _ = chart_image.shape
            side_panel[height // 2 + 10:height // 2 + 10 + chart_height, 10:10 + chart_width] = chart_image

            # Combine frame and side panel
            combined_frame = np.hstack((frame, side_panel))
            out.write(combined_frame)
            video_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
            os.remove(temp_chart_path)

            # Append results
            for au in au_columns:
                new_row = pd.DataFrame({
                    "Frame": [frame_count],
                    "Emotion": [dominant_emotion],
                    "AU": [au],
                    "Confidence": [face[au]],
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update progress
        progress_bar.progress(min((frame_count + 1) / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_count + 1} of {total_frames}")
        frame_count += 1
        summary_stats["Frames Processed"] = frame_count

    cap.release()
    out.release()
    
    # Calculate summary statistics
    if summary_stats["Frames with Faces"] > 0:
        summary_stats["Average Confidence"] /= summary_stats["Frames with Faces"]
        summary_stats["Dominant Emotion"] = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "None"
        summary_stats["Most Active AU"] = max(au_intensities.items(), key=lambda x: x[1])[0] if au_intensities else "None"
        
        # Calculate average Goodnews/Badnews intensities
        num_frames = summary_stats["Frames with Faces"]
        summary_stats["Average Goodnews Intensity"] /= (num_frames * len([au for au in selected_aus if au in happiness_aus]))
        summary_stats["Average Badnews Intensity"] /= (num_frames * len([au for au in selected_aus if au in sadness_aus]))
    
    return results_df, summary_stats

# Streamlit UI
st.title("Facial Expression Analysis App")
tab1, tab2 = st.tabs(["Single Video Analysis", "Multiple Videos Analysis"])

with tab1:
    st.header("Single Video Analysis")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"], key="single_video")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            input_video_path = temp_file.name

        output_video_path = "output_video.mp4"
        
        st.sidebar.header("Customizable Parameters")
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, key="confidence_single")
        selected_aus = st.sidebar.multiselect("Select Action Units (AUs)", list(au_names.keys()), default=list(au_names.keys()), key="aus_single")

        if st.button("Process Video", key="process_single"):
            results_df, summary_stats = process_single_video(input_video_path, output_video_path, confidence_threshold, selected_aus)
            st.video(output_video_path)
            
            # Display summary statistics
            st.subheader("Analysis Summary")
            summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
            st.table(summary_df)
            
            # Download buttons section
            st.subheader("Download Results")
            
            # 1. Download processed video
            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )

            # 2. Download results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Export Results as CSV",
                data=csv,
                file_name="facial_expression_results.csv",
                mime="text/csv",
            )

            # 3. Download summary statistics
            summary_csv = summary_df.to_csv()
            st.download_button(
                label="Export Summary Statistics",
                data=summary_csv,
                file_name="summary_statistics.csv",
                mime="text/csv",
            )

            # 4. Visualizations
            st.subheader("Visualizations")
            
            # Emotion distribution
            if not results_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                results_df['Emotion'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Emotion Distribution Across Frames")
                ax.set_ylabel("Number of Frames")
                st.pyplot(fig)
                st.markdown(get_plot_download_link(fig, "emotion_distribution.png"), unsafe_allow_html=True)
                
                # AU intensity plot
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                results_df.groupby('AU')['Confidence'].mean().sort_values().plot(kind='barh', ax=ax2, color='lightgreen')
                ax2.set_title("Average AU Intensity")
                ax2.set_xlabel("Intensity (0-1)")
                st.pyplot(fig2)
                st.markdown(get_plot_download_link(fig2, "au_intensity.png"), unsafe_allow_html=True)
                
                # AU Time Series with Smoothing
                st.write("### AU Intensity Over Time (Smoothed)")
                
                # Pivot to get AU columns
                au_pivot = results_df.pivot(index='Frame', columns='AU', values='Confidence')
                
                # Apply smoothing
                smoothed_df = au_pivot.copy()
                window_size = min(15, len(smoothed_df)//2 or 1)
                if window_size % 2 == 0:
                    window_size -= 1
                
                for au in smoothed_df.columns:
                    smoothed_df[au] = signal.savgol_filter(
                        smoothed_df[au],
                        window_length=window_size,
                        polyorder=2
                    )
                
                # Create plot with baseline at 0
                fig3 = px.line(smoothed_df, title="AU Intensity (Smoothed)")
                fig3.update_layout(
                    hovermode="x unified",
                    yaxis_title="Intensity",
                    yaxis_range=[0, 1]  # Set y-axis range from 0 to 1
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Download smoothed data
                st.download_button(
                    label="Download Smoothed AU Data",
                    data=smoothed_df.reset_index().to_csv(index=False).encode('utf-8'),
                    file_name="smoothed_au_data.csv",
                    mime="text/csv"
                )

                # Goodnews vs Badnews Comparison
                st.subheader("Goodnews vs Badnews Comparison")
                
                # Filter available AUs
                available_happiness = [au for au in happiness_aus if au in smoothed_df.columns]
                available_sadness = [au for au in sadness_aus if au in smoothed_df.columns]
                
                if available_happiness and available_sadness:
                    # Calculate means
                    happiness_mean = smoothed_df[available_happiness].mean(axis=1).mean()
                    sadness_mean = smoothed_df[available_sadness].mean(axis=1).mean()
                    
                    # Create comparison plot
                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    ax4.bar(['Goodnews AUs', 'Badnews AUs'], 
                           [happiness_mean, sadness_mean],
                           color=['blue', 'orange'])
                    ax4.set_ylabel('Average Intensity')
                    ax4.set_title('Average Goodnews vs Badnews AU Intensity')
                    st.pyplot(fig4)
                    st.markdown(get_plot_download_link(fig4, "goodnews_vs_badnews.png"), unsafe_allow_html=True)
                    
                    # Create detailed comparison
                    st.write("### Detailed AU Comparison")
                    
                    # Prepare data
                    comparison_data = pd.DataFrame({
                        'Type': ['Goodnews'] * len(available_happiness) + ['Badnews'] * len(available_sadness),
                        'AU': available_happiness + available_sadness,
                        'Average Intensity': list(smoothed_df[available_happiness].mean()) + list(smoothed_df[available_sadness].mean())
                    })
                    
                    fig5, ax5 = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=comparison_data, x='AU', y='Average Intensity', hue='Type',
                                palette={'Goodnews': 'blue', 'Badnews': 'orange'}, ax=ax5)
                    ax5.set_title('Detailed AU Comparison')
                    ax5.set_ylim(0, 1)
                    st.pyplot(fig5)
                    st.markdown(get_plot_download_link(fig5, "detailed_au_comparison.png"), unsafe_allow_html=True)
                    
                    # Download comparison data
                    st.download_button(
                        label="Download Comparison Data",
                        data=comparison_data.to_csv(index=False).encode('utf-8'),
                        file_name="goodnews_badnews_comparison.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Insufficient data for Goodnews/Badnews comparison")

        os.remove(input_video_path)

with tab2:
    st.header("Multiple Videos Analysis")
    uploaded_files = st.file_uploader("Upload multiple videos", type=["mp4", "avi"], accept_multiple_files=True, key="multiple_videos")

    if uploaded_files:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting video processing...")

        # Initialize combined results
        all_results = []
        video_names = []
        
        for i, video in enumerate(uploaded_files):
            video_path = os.path.join("temp", video.name)
            with open(video_path, "wb") as f:
                f.write(video.getbuffer())

            # Process video and save results
            out_name = video_path.replace(".mp4", ".csv").replace(".avi", ".csv")
            if not os.path.exists(out_name):
                try:
                    status_text.text(f"Processing {video.name}...")
                    fex = detector.detect_video(video_path)
                    fex['video_name'] = video.name  # Add video name as a column
                    fex.to_csv(out_name)
                    all_results.append(fex)
                    video_names.append(video.name)
                except Exception as e:
                    st.error(f"Error processing {video.name}: {str(e)}")
                    continue

            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success("All videos processed successfully!")
        status_text.empty()

        # Combined analysis section
        if len(all_results) > 1:
            st.subheader("Combined Analysis Across All Videos")
            
            try:
                combined_df = pd.concat(all_results)
                
                # 1. Combined AU Analysis
                au_cols = [col for col in combined_df.columns if col.startswith('AU')]
                if au_cols:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    combined_df.groupby('video_name')[au_cols].mean().T.plot(kind='bar', ax=ax)
                    ax.set_title("Average AU Intensity by Video")
                    ax.set_ylabel("Intensity (0-1)")
                    plt.xticks(rotation=45)
                    plt.legend(title='Video Name', bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)
                    st.markdown(get_plot_download_link(fig, "combined_au_analysis.png"), unsafe_allow_html=True)
                    
                    # Download combined AU data
                    combined_au_data = combined_df.groupby('video_name')[au_cols].mean()
                    st.markdown(get_table_download_link(combined_au_data, "combined_au_data.csv"), unsafe_allow_html=True)
                
                # 2. Combined Emotion Analysis
                emotion_cols = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']
                available_emotions = [col for col in emotion_cols if col in combined_df.columns]
                
                if available_emotions:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    combined_df.groupby('video_name')[available_emotions].mean().plot(kind='bar', ax=ax)
                    ax.set_title("Average Emotion Intensity by Video")
                    ax.set_ylabel("Intensity (0-1)")
                    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)
                    st.markdown(get_plot_download_link(fig, "combined_emotion_analysis.png"), unsafe_allow_html=True)
                    
                    # Download combined emotion data
                    combined_emotion_data = combined_df.groupby('video_name')[available_emotions].mean()
                    st.markdown(get_table_download_link(combined_emotion_data, "combined_emotion_data.csv"), unsafe_allow_html=True)
                    
                    # Emotion radar chart
                    st.write("### Emotion Profile Comparison")
                    fig = px.line_polar(
                        combined_df.groupby('video_name')[available_emotions].mean().reset_index(), 
                        r='happiness', 
                        theta='video_name',
                        line_close=True,
                        title="Emotion Profile Comparison"
                    )
                    st.plotly_chart(fig)
                    
                    # Download plotly figure as HTML
                    plotly_html = fig.to_html()
                    st.download_button(
                        label="Download Interactive Emotion Chart (HTML)",
                        data=plotly_html,
                        file_name="emotion_comparison.html",
                        mime="text/html"
                    )

                # 3. Combined Goodnews vs Badnews Analysis
                st.subheader("Combined Goodnews vs Badnews Analysis")
                
                # Calculate average intensities for each video
                video_stats = []
                for video in all_results:
                    available_happiness = [au for au in happiness_aus if au in video.columns]
                    available_sadness = [au for au in sadness_aus if au in video.columns]
                    
                    if available_happiness and available_sadness:
                        video_name = video['video_name'].iloc[0]
                        happiness_mean = video[available_happiness].mean().mean()
                        sadness_mean = video[available_sadness].mean().mean()
                        
                        video_stats.append({
                            'Video': video_name,
                            'Type': 'Goodnews',
                            'Average Intensity': happiness_mean
                        })
                        video_stats.append({
                            'Video': video_name,
                            'Type': 'Badnews',
                            'Average Intensity': sadness_mean
                        })
                
                if video_stats:
                    comparison_df = pd.DataFrame(video_stats)
                    
                    # Create comparison plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=comparison_df, x='Video', y='Average Intensity', hue='Type',
                                palette={'Goodnews': 'blue', 'Badnews': 'orange'}, ax=ax)
                    ax.set_title('Goodnews vs Badnews Comparison Across Videos')
                    ax.set_ylabel('Average Intensity')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    st.markdown(get_plot_download_link(fig, "combined_goodnews_badnews.png"), unsafe_allow_html=True)
                    
                    # Download comparison data
                    st.download_button(
                        label="Download Goodnews/Badnews Comparison Data",
                        data=comparison_df.to_csv(index=False).encode('utf-8'),
                        file_name="combined_goodnews_badnews.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Insufficient data for combined Goodnews/Badnews comparison")

            except Exception as e:
                st.error(f"Error generating combined analysis: {str(e)}")

        # Display results for each video
        for video in uploaded_files:
            csv_path = os.path.join("temp", video.name.replace(".mp4", ".csv").replace(".avi", ".csv"))
            
            if not os.path.exists(csv_path):
                st.warning(f"No results found for {video.name}")
                continue

            try:
                fex = read_feat(csv_path)
            except Exception as e:
                st.error(f"Error loading results for {video.name}: {str(e)}")
                continue

            st.write(f"## Analysis for: {video.name}")
            
            # 1. AU Detection Bar Chart
            with st.expander("Action Units Analysis"):
                st.write("### Average AU Intensity")
                try:
                    available_aus = [col for col in fex.columns if col.startswith('AU')]
                    if available_aus:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        fex[available_aus].mean().plot(kind='bar', color='steelblue', ax=ax)
                        ax.set_title(f"Action Units Detection - {video.name}")
                        ax.set_ylabel("Intensity (0-1)")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        st.markdown(get_plot_download_link(fig, f"{video.name}_au_analysis.png"), unsafe_allow_html=True)
                        
                        # AU Time Series with Smoothing
                        st.write("### AU Intensity Over Time (Smoothed)")
                        
                        # Apply smoothing
                        smoothed_df = fex[available_aus].copy()
                        window_size = min(15, len(smoothed_df)//2 or 1)
                        if window_size % 2 == 0:
                            window_size -= 1
                        
                        for au in available_aus:
                            smoothed_df[au] = signal.savgol_filter(
                                smoothed_df[au],
                                window_length=window_size,
                                polyorder=2
                            )
                        
                        # Create plot with baseline at 0
                        fig = px.line(smoothed_df, title=f"AU Dynamics - {video.name}")
                        fig.update_layout(
                            hovermode="x unified",
                            yaxis_title="Intensity",
                            yaxis_range=[0, 1]  # Set y-axis range from 0 to 1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download smoothed data
                        st.download_button(
                            label=f"Download Smoothed AU Data for {video.name}",
                            data=smoothed_df.to_csv().encode('utf-8'),
                            file_name=f"{video.name}_au_smoothed.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No AU data available for this video")
                except Exception as e:
                    st.error(f"Error generating AU charts: {str(e)}")

            # 2. Emotions Analysis
            with st.expander("Emotions Analysis"):
                st.write("### Emotion Distribution")
                try:
                    emotion_cols = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']
                    available_emotions = [col for col in emotion_cols if col in fex.columns]
                    
                    if available_emotions:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fex[available_emotions].mean().plot(kind='bar', color='blue', ax=ax)
                        ax.set_title(f"Average Emotion Intensity - {video.name}")
                        ax.set_ylabel("Intensity (0-1)")
                        st.pyplot(fig)
                        st.markdown(get_plot_download_link(fig, f"{video.name}_emotion_distribution.png"), unsafe_allow_html=True)
                        
                        # Emotion Time Series
                        st.write("### Emotion Dynamics Over Time")
                        fig = px.line(fex[available_emotions], title=f"Emotion Timeline - {video.name}")
                        fig.update_layout(yaxis_range=[0, 1])  # Set y-axis range from 0 to 1
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download emotion time series data
                        st.markdown(get_table_download_link(fex[available_emotions], f"{video.name}_emotion_timeseries.csv"), unsafe_allow_html=True)
                    else:
                        st.warning("No emotion data available for this video")
                except Exception as e:
                    st.error(f"Error generating emotion charts: {str(e)}")

            # 3. Goodnews vs Badnews Comparison
            with st.expander("Goodnews vs Badnews Comparison"):
                try:
                    # Get available AUs from our predefined lists
                    available_happiness = [au for au in happiness_aus if au in fex.columns]
                    available_sadness = [au for au in sadness_aus if au in fex.columns]
                    
                    if available_happiness and available_sadness:
                        # Calculate means for each group
                        happiness_mean = fex[available_happiness].mean().mean()
                        sadness_mean = fex[available_sadness].mean().mean()
                        
                        # Create comparison plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.bar(['Goodnews AUs', 'Badnews AUs'], 
                              [happiness_mean, sadness_mean],
                              color=['blue', 'orange'])
                        ax.set_ylabel('Average Intensity')
                        ax.set_title(f'Goodnews vs Badnews Comparison - {video.name}')
                        st.pyplot(fig)
                        st.markdown(get_plot_download_link(fig, f"{video.name}_goodnews_vs_badnews.png"), unsafe_allow_html=True)
                        
                        # Create detailed comparison
                        st.write("### Detailed AU Comparison")
                        
                        # Prepare data
                        comparison_data = pd.DataFrame({
                            'Type': ['Goodnews'] * len(available_happiness) + ['Badnews'] * len(available_sadness),
                            'AU': available_happiness + available_sadness,
                            'Average Intensity': list(fex[available_happiness].mean()) + list(fex[available_sadness].mean())
                        })
                        
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=comparison_data, x='AU', y='Average Intensity', hue='Type',
                                    palette={'Goodnews': 'blue', 'Badnews': 'orange'}, ax=ax2)
                        ax2.set_title(f'Detailed AU Comparison - {video.name}')
                        ax2.set_ylim(0, 1)
                        st.pyplot(fig2)
                        st.markdown(get_plot_download_link(fig2, f"{video.name}_detailed_au_comparison.png"), unsafe_allow_html=True)
                        
                        # Download comparison data
                        st.download_button(
                            label=f"Download Comparison Data for {video.name}",
                            data=comparison_data.to_csv(index=False).encode('utf-8'),
                            file_name=f"{video.name}_goodnews_badnews_comparison.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Insufficient data for Goodnews/Badnews comparison")
                except Exception as e:
                    st.error(f"Error generating comparison chart: {str(e)}")

            # Download buttons
            st.download_button(
                label=f"Download {video.name} Results (CSV)",
                data=open(csv_path, 'rb').read(),
                file_name=f"{video.name}_results.csv",
                mime="text/csv"
            )
            
            st.write("---")

if st.button("Clear All Temporary Files"):
    for file in os.listdir("temp"):
        file_path = os.path.join("temp", file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error deleting {file_path}: {e}")
    st.success("All temporary files cleared!")