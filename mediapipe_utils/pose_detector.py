"""
DeepPose 2.0 — MediaPipe Pose Detector
========================================
Wraps MediaPipe Pose for landmark extraction, skeleton visualization,
and feature vector generation for tabular classification.
"""

import os
import sys
from typing import Optional, List

import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PoseDetector:
    """MediaPipe Pose wrapper for landmark extraction and visualization.
    
    Extracts 33 body landmarks with (x, y, z, visibility) from BGR frames,
    draws skeleton overlay, and produces flat feature vectors for tabular models.
    
    Args:
        static_image_mode: If True, treats each frame independently (default: False)
        min_detection_confidence: Minimum confidence for initial detection (default: 0.5)
        min_tracking_confidence: Minimum confidence for landmark tracking (default: 0.5)
    
    Example:
        >>> detector = PoseDetector()
        >>> landmarks = detector.detect(frame)
        >>> if landmarks:
        ...     annotated = detector.draw_skeleton(frame, landmarks)
        ...     features = detector.extract_feature_vector(landmarks)
    """
    
    def __init__(self, static_image_mode: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, frame: np.ndarray) -> Optional[List]:
        """Extract 33 pose landmarks from a BGR frame.
        
        Processes the frame through MediaPipe Pose and returns landmarks
        if a person is detected.
        
        Args:
            frame: BGR numpy array of shape (H, W, 3)
        
        Returns:
            List of 33 landmark objects with (x, y, z, visibility) attributes,
            or None if no pose is detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def draw_skeleton(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw skeleton overlay on frame using MediaPipe drawing utilities.
        
        Args:
            frame: BGR numpy array to draw on (will be copied)
            landmarks: List of 33 MediaPipe landmarks
        
        Returns:
            Copy of frame with skeleton overlay drawn
        """
        annotated_frame = frame.copy()
        
        # Create a NormalizedLandmarkList for drawing
        landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for lm in landmarks:
            landmark_proto = landmark_list.landmark.add()
            landmark_proto.x = lm.x
            landmark_proto.y = lm.y
            landmark_proto.z = lm.z
            landmark_proto.visibility = lm.visibility
        
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return annotated_frame
    
    def extract_feature_vector(self, landmarks) -> np.ndarray:
        """Extract 132-dimensional feature vector from landmarks.
        
        Flattens all 33 landmarks × 4 values (x, y, z, visibility) into
        a single numpy array suitable for tabular classification models.
        
        Args:
            landmarks: List of 33 MediaPipe landmarks
        
        Returns:
            Numpy array of shape (132,) containing flattened landmark data
        """
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        return np.array(features, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.pose.close()
        except:
            pass
