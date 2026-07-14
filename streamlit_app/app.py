"""
DeepPose 2.0 — Streamlit Dashboard
====================================
Multi-page web interface for AI-powered yoga pose recognition and correction.
Pages: Home, Image Upload, Live Camera, Model Performance.
"""

import os
import sys
import time
import json

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.inference import PoseInferenceEngine
from mediapipe_utils.pose_detector import PoseDetector
from scoring.angle_utils import AngleCalculator
from scoring.posture_score import PostureScorer, IDEAL_ANGLES
from scoring.feedback_engine import FeedbackEngine
from voice.voice_feedback import VoiceFeedback
from models.session import (
    UserSession, SessionState, update_session_state, get_smoothed_prediction
)


# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================
# Sidebar Navigation
# ========================
def render_sidebar():
    """Render navigation sidebar."""
    st.sidebar.title("🧘 DeepPose 2.0")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📸 Image Upload", "🎥 Live Camera", "📊 Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**DeepPose 2.0**\n\n"
        "AI-Powered Yoga Pose\n"
        "Recognition & Correction"
    )
    
    return page


# ========================
# Home Page
# ========================
def render_home():
    """Render the Home page with project information."""
    st.title("🧘 DeepPose 2.0")
    st.subheader("AI-Powered Yoga Pose Recognition & Correction System")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What it does")
        st.markdown("""
        - **Recognizes** 11 yoga poses in real-time
        - **Scores** your posture quality (0-100)
        - **Provides** corrective feedback
        - **Speaks** guidance using voice feedback
        - **Visualizes** your body skeleton
        """)
        
        st.markdown("### 🛠️ Technology")
        st.markdown("""
        - ResNet18 (Transfer Learning)
        - MediaPipe Pose (33 Landmarks)
        - OpenCV (Real-time Video)
        - pyttsx3 (Voice Feedback)
        - Streamlit (Web Dashboard)
        """)
    
    with col2:
        st.markdown("### 🧘‍♀️ Supported Poses")
        poses = config.CLASSES
        for i, pose in enumerate(poses, 1):
            st.markdown(f"{i}. {pose.replace('_', ' ')}")
    
    st.markdown("---")
    
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Image Upload** — Upload a yoga pose photo for instant analysis
    2. **Live Camera** — Get real-time feedback while practicing
    3. **Model Performance** — View training metrics and accuracy
    """)
    
    st.markdown("---")
    st.markdown("### 📊 System Overview")
    st.markdown("""
    ```
    Camera/Image → ResNet18 Classification → Pose Name + Confidence
                → MediaPipe Landmarks → Skeleton Visualization
                → Angle Calculation → Posture Score (0-100)
                → Feedback Engine → Corrective Instructions
                → Voice Engine → Spoken Guidance
    ```
    """)


# ========================
# Image Upload Page
# ========================
def render_image_upload():
    """Render the Image Upload page for single-image analysis."""
    st.title("📸 Image Upload — Pose Analysis")
    st.markdown("Upload a yoga pose image to get instant pose classification, "
                "posture score, and corrective feedback.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose a yoga pose image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear image of someone performing a yoga pose"
    )
    
    if uploaded_file is not None:
        # Validate file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.MAX_UPLOAD_SIZE_MB:
            st.error(f"❌ File too large ({file_size_mb:.1f} MB). "
                    f"Maximum allowed: {config.MAX_UPLOAD_SIZE_MB} MB")
            return
        
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Initialize components
        try:
            engine = get_inference_engine()
            detector = get_pose_detector()
            angle_calc = AngleCalculator()
            scorer = PostureScorer()
            feedback_engine = FeedbackEngine()
        except FileNotFoundError as e:
            st.error(f"❌ Model not found. Please train the model first.\n\n{e}")
            return
        
        # Run pipeline
        with st.spinner("Analyzing pose..."):
            # Classification
            pose_name, confidence = engine.predict(frame_bgr)
            
            # Landmark detection
            landmarks = detector.detect(frame_bgr)
            
            # Results layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Input Image")
                if landmarks:
                    annotated = detector.draw_skeleton(frame_bgr, landmarks)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Skeleton Overlay", use_column_width=True)
                else:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("### 📊 Analysis Results")
                
                # Pose prediction
                st.metric("Pose Detected", pose_name.replace("_", " "))
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                if landmarks:
                    # Angle calculation
                    angles = angle_calc.get_all_angles(landmarks)
                    
                    if angles:
                        # Posture scoring
                        score, deviations = scorer.score(pose_name, angles)
                        ideal = IDEAL_ANGLES.get(pose_name, {})
                        
                        # Display score with color
                        if score >= 80:
                            st.success(f"🎯 Posture Score: **{score:.0f}/100** — Excellent!")
                        elif score >= 60:
                            st.warning(f"🎯 Posture Score: **{score:.0f}/100** — Good, minor adjustments needed")
                        else:
                            st.error(f"🎯 Posture Score: **{score:.0f}/100** — Needs improvement")
                        
                        # Feedback
                        feedback = feedback_engine.generate_feedback(
                            pose_name, deviations, angles, ideal
                        )
                        
                        if feedback:
                            st.markdown("### 💡 Corrective Feedback")
                            for i, instruction in enumerate(feedback, 1):
                                st.markdown(f"**{i}.** {instruction}")
                        else:
                            st.success("✅ Great form! No corrections needed.")
                        
                        # Joint angles details
                        with st.expander("🔍 Joint Angle Details"):
                            for joint, angle in angles.items():
                                ideal_val = ideal.get(joint, "N/A")
                                dev = deviations.get(joint, 0)
                                status = "✅" if dev < config.DEVIATION_THRESHOLD else "⚠️"
                                st.markdown(
                                    f"{status} **{joint}**: {angle:.1f}° "
                                    f"(ideal: {ideal_val}°, deviation: {dev:.1f}°)"
                                )
                    else:
                        st.warning("⚠️ Could not calculate angles — landmarks partially visible")
                else:
                    st.warning("⚠️ No pose detected in the image. "
                             "Please upload a clearer image with a visible person.")


# ========================
# Live Camera Page
# ========================
def render_live_camera():
    """Render the Live Camera page for real-time pose detection."""
    st.title("🎥 Live Camera — Real-Time Pose Detection")
    st.markdown("Real-time yoga pose detection with posture scoring and feedback.")
    
    st.markdown("---")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        voice_enabled = st.checkbox("🔊 Voice Feedback", value=False)
    with col2:
        show_skeleton = st.checkbox("🦴 Show Skeleton", value=True)
    with col3:
        show_angles = st.checkbox("📐 Show Angles", value=False)
    
    st.markdown("---")
    
    # Camera input using Streamlit's built-in camera
    camera_image = st.camera_input("Take a photo for pose analysis")
    
    if camera_image is not None:
        # Process the camera frame
        image = Image.open(camera_image).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        try:
            engine = get_inference_engine()
            detector = get_pose_detector()
            angle_calc = AngleCalculator()
            scorer = PostureScorer()
            feedback_engine = FeedbackEngine()
        except FileNotFoundError as e:
            st.error(f"❌ Model not found: {e}")
            return
        
        # Initialize session state
        if 'user_session' not in st.session_state:
            st.session_state.user_session = UserSession()
        
        # Run pipeline
        pose_name, confidence = engine.predict(frame_bgr)
        landmarks = detector.detect(frame_bgr)
        
        # Update state machine
        session = st.session_state.user_session
        session = update_session_state(session, pose_name, confidence)
        st.session_state.user_session = session
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_frame = frame_bgr.copy()
            
            if landmarks and show_skeleton:
                display_frame = detector.draw_skeleton(display_frame, landmarks)
            
            # Add text overlay
            cv2.putText(display_frame, f"Pose: {pose_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {confidence*100:.1f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"State: {session.state.value}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            st.image(display_rgb, caption="Live Analysis", use_column_width=True)
        
        with col2:
            st.markdown("### 📊 Real-Time Stats")
            st.metric("Pose", pose_name.replace("_", " "))
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("State", session.state.value.upper())
            
            if landmarks and session.state == SessionState.HOLDING_POSE:
                angles = angle_calc.get_all_angles(landmarks)
                if angles:
                    score, deviations = scorer.score(pose_name, angles)
                    ideal = IDEAL_ANGLES.get(pose_name, {})
                    
                    st.metric("Posture Score", f"{score:.0f}/100")
                    
                    feedback = feedback_engine.generate_feedback(
                        pose_name, deviations, angles, ideal
                    )
                    
                    if feedback:
                        st.markdown("### 💡 Feedback")
                        for instruction in feedback:
                            st.markdown(f"• {instruction}")
                        
                        # Voice feedback
                        if voice_enabled:
                            if 'voice' not in st.session_state:
                                st.session_state.voice = VoiceFeedback()
                            if st.session_state.voice.is_available:
                                st.session_state.voice.speak(feedback[0])
                                st.markdown("🔊 *Voice active*")
                            else:
                                st.markdown("🔇 *Voice unavailable*")
                    else:
                        st.success("✅ Perfect form!")
                    
                    if show_angles:
                        with st.expander("📐 Angles"):
                            for joint, angle in angles.items():
                                st.text(f"{joint}: {angle:.1f}°")
            else:
                if session.state == SessionState.TRANSITIONING:
                    st.info("🔄 Transitioning... hold your pose")
                elif session.state == SessionState.IDLE:
                    st.info("🧘 Strike a pose to begin")
    else:
        st.info("📷 Click 'Take a photo' above to capture your pose.\n\n"
               "**Tip:** For best results, ensure your full body is visible "
               "with good lighting.")


# ========================
# Model Performance Page
# ========================
def render_model_performance():
    """Render the Model Performance page with training metrics."""
    st.title("📊 Model Performance")
    st.markdown("Training metrics and evaluation results for the ResNet18 pose classifier.")
    
    st.markdown("---")
    
    # Check if output files exist
    confusion_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    accuracy_path = os.path.join(config.OUTPUT_DIR, "accuracy_curve.png")
    loss_path = os.path.join(config.OUTPUT_DIR, "loss_curve.png")
    
    has_outputs = any(os.path.exists(p) for p in [confusion_path, accuracy_path, loss_path])
    
    if not has_outputs:
        st.warning(
            "⚠️ **Model has not been trained yet.**\n\n"
            "Run the training pipeline to generate metrics:\n"
            "```bash\n"
            "python models/train.py\n"
            "```"
        )
        
        # Show expected metrics as placeholder
        st.markdown("### Expected Metrics After Training")
        st.markdown("""
        | Metric | Target |
        |--------|--------|
        | Overall Accuracy | >75% |
        | Per-class F1 Score | >70% |
        | Inference Speed (CPU) | <50ms |
        | Inference Speed (GPU) | <20ms |
        """)
        return
    
    # Display training curves
    st.markdown("### 📈 Training Curves")
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(loss_path):
            st.image(loss_path, caption="Training & Validation Loss")
        else:
            st.info("Loss curve not available")
    
    with col2:
        if os.path.exists(accuracy_path):
            st.image(accuracy_path, caption="Training & Validation Accuracy")
        else:
            st.info("Accuracy curve not available")
    
    # Confusion Matrix
    st.markdown("### 🔢 Confusion Matrix")
    if os.path.exists(confusion_path):
        st.image(confusion_path, caption="Confusion Matrix — All 11 Classes")
    else:
        st.info("Confusion matrix not available")
    
    # Model info
    st.markdown("### ℹ️ Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Architecture**: ResNet18 (Transfer Learning)
        - **Pretrained**: ImageNet (IMAGENET1K_V1)
        - **Backbone**: Frozen (no fine-tuning)
        - **Classifier**: Linear(512 → 11)
        """)
    with col2:
        st.markdown(f"""
        - **Image Size**: {config.IMG_SIZE}×{config.IMG_SIZE}
        - **Batch Size**: {config.BATCH_SIZE}
        - **Learning Rate**: {config.LEARNING_RATE}
        - **Early Stopping**: Patience = {config.PATIENCE}
        """)


# ========================
# Component Caching
# ========================
@st.cache_resource
def get_inference_engine():
    """Load inference engine (cached across Streamlit reruns)."""
    return PoseInferenceEngine()


@st.cache_resource
def get_pose_detector():
    """Load pose detector (cached across Streamlit reruns)."""
    return PoseDetector(static_image_mode=True)


# ========================
# Main App Entry Point
# ========================
def main():
    """Main application entry point."""
    page = render_sidebar()
    
    if page == "🏠 Home":
        render_home()
    elif page == "📸 Image Upload":
        render_image_upload()
    elif page == "🎥 Live Camera":
        render_live_camera()
    elif page == "📊 Model Performance":
        render_model_performance()


if __name__ == "__main__":
    main()
