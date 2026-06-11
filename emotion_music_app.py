"""
Emotion-Based Music Recommendation System - Streamlit App
Author: FC211033 Sahan
Date: October 2025

A real-time emotion detection system that recommends music from YouTube
based on detected emotions using computer vision and deep learning.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import time
from collections import Counter, deque
import yt_dlp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Emotion Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .song-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        border-left-color: #764ba2;
    }
    .song-title {
        color: #2c3e50;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .song-info {
        color: #555;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CONFIG = {
    'model_path': 'notebooks/FC211033_Sahan/models/best_resnet_model_mac.pth',
    'cascade_path': 'haarcascade_frontalface_default.xml',
    'image_size': 64,
    'emotions': ['angry', 'happy', 'neutral', 'sad', 'surprise'],
    'num_classes': 5,
}

EMOTION_MUSIC_MAPPING = {
    'angry': {
        'search_queries': ['rock music official', 'metal playlist hits', 'punk rock official'],
        'description': '🔥 Intense, high-energy music',
        'genres': 'Rock, Metal, Punk',
        'color': '#FF4444',
        'emoji': '😠'
    },
    'happy': {
        'search_queries': ['happy pop music official', 'upbeat songs hits', 'feel good music official'],
        'description': '😊 Upbeat, feel-good music',
        'genres': 'Pop, Dance, Funk',
        'color': '#FFD700',
        'emoji': '😊'
    },
    'neutral': {
        'search_queries': ['chill music official', 'lofi hip hop popular', 'relaxing music hits'],
        'description': '😌 Calm, balanced music',
        'genres': 'Chill, Ambient, Lo-fi',
        'color': '#808080',
        'emoji': '😐'
    },
    'sad': {
        'search_queries': ['sad songs official', 'emotional music hits', 'ballad official'],
        'description': '😢 Melancholic, emotional music',
        'genres': 'Ballad, Piano, Soul',
        'color': '#4169E1',
        'emoji': '😢'
    },
    'surprise': {
        'search_queries': ['edm music official', 'electronic hits popular', 'dance music official'],
        'description': '🎉 Energetic, surprising music',
        'genres': 'EDM, Electronic, Dubstep',
        'color': '#FF69B4',
        'emoji': '😲'
    }
}

# Emotion Detection Model
class EmotionResNet(nn.Module):
    """ResNet-18 for emotion recognition."""
    
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(EmotionResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# YouTube Music Recommender
class YouTubeMusicRecommender:
    """Music recommendation system using YouTube."""
    
    def __init__(self):
        self.current_recommendations = []
        self.last_emotion = None
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }
    
    def get_recommendations(self, emotion, limit=8):
        """Get music recommendations from YouTube based on emotion."""
        try:
            mapping = EMOTION_MUSIC_MAPPING[emotion]
            recommendations = []
            query = mapping['search_queries'][0]
            
            # Enhanced search with sorting by view count
            search_url = f"ytsearch{limit*2}:{query} official"  # Search more to filter better
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                result = ydl.extract_info(search_url, download=False)
                
                if 'entries' in result:
                    # Collect all videos with view counts
                    videos_with_views = []
                    for video in result['entries']:
                        if video and video.get('view_count'):
                            videos_with_views.append(video)
                    
                    # Sort by view count (most popular first)
                    videos_with_views.sort(key=lambda x: x.get('view_count', 0), reverse=True)
                    
                    # Take top results
                    for video in videos_with_views[:limit]:
                        track_info = {
                            'title': video.get('title', 'Unknown'),
                            'channel': video.get('uploader', 'Unknown'),
                            'duration': self._format_duration(video.get('duration', 0)),
                            'views': self._format_views(video.get('view_count', 0)),
                            'url': f"https://www.youtube.com/watch?v={video['id']}",
                            'video_id': video['id'],
                            'thumbnail': video.get('thumbnail', None)
                        }
                        recommendations.append(track_info)
            
            self.current_recommendations = recommendations
            self.last_emotion = emotion
            return recommendations
        
        except Exception as e:
            st.error(f"Error getting recommendations: {e}")
            return []
    
    def _format_duration(self, seconds):
        if not seconds:
            return "N/A"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    def _format_views(self, view_count):
        if not view_count:
            return "N/A"
        if view_count >= 1_000_000:
            return f"{view_count / 1_000_000:.1f}M"
        elif view_count >= 1_000:
            return f"{view_count / 1_000:.1f}K"
        else:
            return str(view_count)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = Counter()

@st.cache_resource
def load_model():
    """Load the emotion detection model."""
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = EmotionResNet(CONFIG['num_classes']).to(device)
        
        if os.path.exists(CONFIG['model_path']):
            checkpoint = torch.load(CONFIG['model_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, device
        else:
            st.error(f"Model not found at {CONFIG['model_path']}")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def load_face_detector():
    """Load the face detector."""
    face_cascade = cv2.CascadeClassifier(CONFIG['cascade_path'])
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

@st.cache_resource
def get_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

@st.cache_resource
def get_recommender():
    """Get the music recommender."""
    return YouTubeMusicRecommender()

def detect_emotion(frame, model, face_cascade, transform, device):
    """Detect emotion from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=8, minSize=(80, 80)
    )
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (CONFIG['image_size'], CONFIG['image_size']))
        pil_image = Image.fromarray(face_resized)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion_idx = predicted.item()
            emotion = CONFIG['emotions'][emotion_idx]
            confidence_score = confidence.item()
            
            return emotion, confidence_score, (x, y, w, h)
    
    return None, 0.0, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Emotion-Based Music Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time emotion detection meets personalized music recommendations")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/musical-notes.png", width=100)
        st.title("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.6,
            step=0.05,
            help="Minimum confidence for emotion detection"
        )
        
        buffer_size = st.slider(
            "Emotion Buffer Size",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Number of frames to track for emotion stability"
        )
        
        update_interval = st.slider(
            "Update Interval (seconds)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="How often to update music recommendations"
        )
        
        num_recommendations = st.slider(
            "Number of Songs",
            min_value=4,
            max_value=12,
            value=8,
            step=2,
            help="Number of songs to recommend"
        )
        
        st.divider()
        
        st.subheader("📊 Statistics")
        if st.session_state.emotion_counts:
            total = sum(st.session_state.emotion_counts.values())
            st.metric("Total Detections", total)
            
            if st.session_state.current_emotion:
                emoji = EMOTION_MUSIC_MAPPING[st.session_state.current_emotion]['emoji']
                st.metric("Current Mood", f"{emoji} {st.session_state.current_emotion.title()}")
        
        st.divider()
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.emotion_history = []
            st.session_state.emotion_counts = Counter()
            st.rerun()
        
        st.divider()
        st.markdown("### 💡 About")
        st.info(
            "This app uses deep learning to detect your emotions in real-time "
            "and recommends music from YouTube that matches your mood. "
            "No API keys required!"
        )
        
        st.markdown("**Author:** FC211033 Sahan")
        st.markdown("**Model:** ResNet-18")
        st.markdown("**Music Source:** YouTube")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "🎵 Recommendations", "📈 Analytics"])
    
    # Tab 1: Live Detection
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Webcam Feed")
            camera_placeholder = st.empty()
            status_placeholder = st.empty()
        
        with col2:
            st.subheader("🎯 Current Status")
            emotion_placeholder = st.empty()
            confidence_placeholder = st.empty()
            
            st.divider()
            
            st.subheader("🎮 Controls")
            start_col, stop_col = st.columns(2)
            
            with start_col:
                if st.button("▶️ Start Detection", disabled=st.session_state.detection_active):
                    st.session_state.detection_active = True
                    st.rerun()
            
            with stop_col:
                if st.button("⏹️ Stop Detection", disabled=not st.session_state.detection_active):
                    st.session_state.detection_active = False
                    st.rerun()
        
        # Detection loop
        if st.session_state.detection_active:
            # Load resources
            model, device = load_model()
            face_cascade = load_face_detector()
            transform = get_transform()
            recommender = get_recommender()
            
            if model is None:
                st.error("❌ Failed to load model. Please check the model path.")
                st.session_state.detection_active = False
                st.stop()
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam")
                st.session_state.detection_active = False
                st.stop()
            
            emotion_buffer = deque(maxlen=buffer_size)
            last_update_time = time.time()
            frame_count = 0
            
            status_placeholder.success("🟢 Detection Active")
            
            try:
                while st.session_state.detection_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    frame_count += 1
                    
                    # Detect emotion every few frames
                    if frame_count % 2 == 0:
                        emotion, confidence, face_coords = detect_emotion(
                            frame, model, face_cascade, transform, device
                        )
                        
                        if emotion and confidence >= confidence_threshold:
                            emotion_buffer.append(emotion)
                            st.session_state.emotion_counts[emotion] += 1
                            
                            # Draw rectangle and label
                            if face_coords:
                                x, y, w, h = face_coords
                                color = (0, 255, 0)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label = f"{emotion.upper()} ({confidence*100:.0f}%)"
                                cv2.putText(frame, label, (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Update current emotion
                    if emotion_buffer:
                        dominant_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                        st.session_state.current_emotion = dominant_emotion
                        
                        # Update recommendations periodically
                        current_time = time.time()
                        if current_time - last_update_time >= update_interval:
                            recommendations = recommender.get_recommendations(
                                dominant_emotion, limit=num_recommendations
                            )
                            st.session_state.recommendations = recommendations
                            last_update_time = current_time
                        
                        # Display emotion
                        mapping = EMOTION_MUSIC_MAPPING[dominant_emotion]
                        emotion_placeholder.markdown(
                            f"<div style='background-color: {mapping['color']}; padding: 20px; "
                            f"border-radius: 10px; text-align: center;'>"
                            f"<h2 style='color: white; margin: 0;'>{mapping['emoji']} "
                            f"{dominant_emotion.upper()}</h2>"
                            f"<p style='color: white; margin: 5px 0 0 0;'>{mapping['description']}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        if confidence > 0:
                            confidence_placeholder.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.03)  # ~30 FPS
            
            finally:
                cap.release()
                status_placeholder.warning("🟡 Detection Stopped")
    
    # Tab 2: Recommendations
    with tab2:
        st.subheader("🎵 Music Recommendations")
        
        if st.session_state.current_emotion and st.session_state.recommendations:
            emotion = st.session_state.current_emotion
            mapping = EMOTION_MUSIC_MAPPING[emotion]
            
            st.markdown(
                f"<div style='background: linear-gradient(135deg, {mapping['color']}22 0%, {mapping['color']}44 100%); "
                f"padding: 25px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
                f"<h2 style='margin: 0; color: #2c3e50;'>{mapping['emoji']} {emotion.upper()} Mood</h2>"
                f"<p style='margin: 10px 0 5px 0; font-size: 1.1rem; color: #34495e;'>{mapping['description']}</p>"
                f"<p style='margin: 0; color: #7f8c8d;'><strong>Genres:</strong> {mapping['genres']}</p>"
                f"<p style='margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;'>🎯 Showing most popular official tracks</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Display recommendations in grid
            cols = st.columns(2)
            
            for idx, track in enumerate(st.session_state.recommendations):
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(
                            f"<div class='song-card'>"
                            f"<div class='song-title'>🎵 {track['title']}</div>"
                            f"<div class='song-info'>👤 {track['channel']}</div>"
                            f"<div class='song-info'>⏱️ {track['duration']} | 👁️ {track['views']} views</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # YouTube embed
                        st.video(track['url'])
                        st.markdown("---")
        else:
            st.info("👆 Start detection in the 'Live Detection' tab to see recommendations!")
            
            # Show example for all emotions
            st.subheader("🎨 Emotion Palette")
            cols = st.columns(5)
            
            for idx, emotion in enumerate(CONFIG['emotions']):
                with cols[idx]:
                    mapping = EMOTION_MUSIC_MAPPING[emotion]
                    if st.button(f"{mapping['emoji']}\n{emotion.title()}", key=f"test_{emotion}"):
                        recommender = get_recommender()
                        recs = recommender.get_recommendations(emotion, limit=num_recommendations)
                        st.session_state.recommendations = recs
                        st.session_state.current_emotion = emotion
                        st.rerun()
    
    # Tab 3: Analytics
    with tab3:
        st.subheader("📈 Emotion Analytics")
        
        if st.session_state.emotion_counts:
            # Pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution")
                df = pd.DataFrame({
                    'Emotion': list(st.session_state.emotion_counts.keys()),
                    'Count': list(st.session_state.emotion_counts.values())
                })
                
                colors = [EMOTION_MUSIC_MAPPING[e]['color'] for e in df['Emotion']]
                
                fig = px.pie(
                    df,
                    values='Count',
                    names='Emotion',
                    title='Emotion Distribution',
                    color_discrete_sequence=colors
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Frequency")
                fig = px.bar(
                    df,
                    x='Emotion',
                    y='Count',
                    title='Emotion Frequency',
                    color='Emotion',
                    color_discrete_sequence=colors
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.divider()
            st.subheader("📊 Detailed Statistics")
            
            cols = st.columns(5)
            for idx, (emotion, count) in enumerate(st.session_state.emotion_counts.most_common()):
                with cols[idx % 5]:
                    mapping = EMOTION_MUSIC_MAPPING[emotion]
                    percentage = (count / sum(st.session_state.emotion_counts.values())) * 100
                    st.markdown(
                        f"<div style='background-color: {mapping['color']}22; padding: 15px; "
                        f"border-radius: 10px; text-align: center; border: 2px solid {mapping['color']};'>"
                        f"<h3>{mapping['emoji']}</h3>"
                        f"<p><strong>{emotion.title()}</strong></p>"
                        f"<h4>{count}</h4>"
                        f"<p>{percentage:.1f}%</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("📊 No data yet. Start detection to see analytics!")
            st.image("https://img.icons8.com/clouds/400/000000/bar-chart.png", width=200)

if __name__ == "__main__":
    main()
