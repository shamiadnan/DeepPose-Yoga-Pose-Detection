"""
DeepPose 2.0 — Angle Calculator
=================================
Computes joint angles from MediaPipe landmarks using vector dot product formula.
Provides biomechanical angle measurements for posture scoring.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# MediaPipe Pose landmark indices
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}

# Joint definitions: (point_a, vertex, point_c) for angle calculation
JOINT_DEFINITIONS = {
    'left_knee': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    'right_knee': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    'left_hip': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
    'right_hip': ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    'left_elbow': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
}

# Minimum visibility threshold for landmark inclusion
VISIBILITY_THRESHOLD = 0.3


class AngleCalculator:
    """Computes joint angles from MediaPipe pose landmarks.
    
    Uses the vector dot product formula to calculate angles at joint vertices.
    Handles degenerate cases (coincident points) and low-visibility landmarks.
    
    Formula:
        angle = arccos(dot(BA, BC) / (|BA| * |BC| + epsilon))
    
    Example:
        >>> calc = AngleCalculator()
        >>> angles = calc.get_all_angles(landmarks)
        >>> print(angles)  # {'left_knee': 170.2, 'right_knee': 92.5, ...}
    """
    
    def __init__(self, visibility_threshold: float = VISIBILITY_THRESHOLD):
        self.visibility_threshold = visibility_threshold
    
    def calculate_angle(self, point_a: np.ndarray, point_b: np.ndarray,
                        point_c: np.ndarray) -> float:
        """Calculate angle at point_b formed by vectors BA and BC.
        
        Uses the dot product formula:
            cos(angle) = dot(BA, BC) / (|BA| * |BC|)
        
        Args:
            point_a: First endpoint as numpy array (2D or 3D)
            point_b: Vertex point where angle is measured
            point_c: Second endpoint as numpy array (2D or 3D)
        
        Returns:
            Angle in degrees, clamped to [0.0, 180.0]
        
        Note:
            Uses epsilon (1e-8) to handle coincident points without division by zero.
        """
        # Compute vectors from vertex (B) to endpoints
        vector_ba = point_a - point_b
        vector_bc = point_c - point_b
        
        # Compute dot product
        dot_product = np.dot(vector_ba, vector_bc)
        
        # Compute magnitudes with epsilon for numerical stability
        magnitude_ba = np.linalg.norm(vector_ba)
        magnitude_bc = np.linalg.norm(vector_bc)
        
        # Avoid division by zero for coincident points
        denominator = magnitude_ba * magnitude_bc + 1e-8
        
        # Compute cosine and clamp to valid range [-1, 1]
        cos_angle = dot_product / denominator
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle_degrees = np.degrees(np.arccos(cos_angle))
        
        return float(angle_degrees)
    
    def _get_landmark_point(self, landmarks, index: int) -> Optional[np.ndarray]:
        """Extract (x, y, z) point from landmark by index.
        
        Args:
            landmarks: MediaPipe landmarks list (33 landmarks)
            index: Landmark index
        
        Returns:
            Numpy array [x, y, z] or None if visibility below threshold
        """
        lm = landmarks[index]
        if lm.visibility < self.visibility_threshold:
            return None
        return np.array([lm.x, lm.y, lm.z])
    
    def _calculate_shoulder_alignment(self, landmarks) -> Optional[float]:
        """Calculate shoulder alignment angle relative to horizontal.
        
        Measures how level the shoulders are. 0° = perfectly level.
        
        Args:
            landmarks: MediaPipe landmarks list
        
        Returns:
            Angle in degrees [0, 90] or None if landmarks not visible
        """
        left_shoulder = self._get_landmark_point(
            landmarks, LANDMARK_INDICES['LEFT_SHOULDER']
        )
        right_shoulder = self._get_landmark_point(
            landmarks, LANDMARK_INDICES['RIGHT_SHOULDER']
        )
        
        if left_shoulder is None or right_shoulder is None:
            return None
        
        # Vector between shoulders (using x, y only)
        shoulder_vector = right_shoulder[:2] - left_shoulder[:2]
        
        # Horizontal reference vector
        horizontal = np.array([1.0, 0.0])
        
        # Angle between shoulder line and horizontal
        dot = np.dot(shoulder_vector, horizontal)
        mag = np.linalg.norm(shoulder_vector) + 1e-8
        cos_angle = np.clip(dot / mag, -1.0, 1.0)
        angle = np.abs(np.degrees(np.arccos(cos_angle)))
        
        # Normalize to [0, 90] — 0 means level
        if angle > 90:
            angle = 180 - angle
        
        return float(angle)
    
    def get_all_angles(self, landmarks) -> Dict[str, float]:
        """Compute all 7 joint angles from MediaPipe landmarks.
        
        Calculates:
        - left_knee, right_knee (hip-knee-ankle)
        - left_hip, right_hip (shoulder-hip-knee)
        - left_elbow, right_elbow (shoulder-elbow-wrist)
        - shoulder_alignment (shoulder line vs horizontal)
        
        Joints with low-visibility landmarks are excluded from output.
        
        Args:
            landmarks: List of 33 MediaPipe pose landmarks
        
        Returns:
            Dictionary mapping joint names to angle values in degrees [0, 180].
            Only includes joints where all required landmarks are visible.
            
        Note:
            Computation time target: < 1ms (pure numpy vectorized operations)
        """
        angles = {}
        
        # Calculate standard joint angles
        for joint_name, (name_a, name_b, name_c) in JOINT_DEFINITIONS.items():
            idx_a = LANDMARK_INDICES[name_a]
            idx_b = LANDMARK_INDICES[name_b]
            idx_c = LANDMARK_INDICES[name_c]
            
            point_a = self._get_landmark_point(landmarks, idx_a)
            point_b = self._get_landmark_point(landmarks, idx_b)
            point_c = self._get_landmark_point(landmarks, idx_c)
            
            # Skip if any landmark has low visibility
            if point_a is None or point_b is None or point_c is None:
                continue
            
            angles[joint_name] = self.calculate_angle(point_a, point_b, point_c)
        
        # Calculate shoulder alignment
        shoulder_angle = self._calculate_shoulder_alignment(landmarks)
        if shoulder_angle is not None:
            angles['shoulder_alignment'] = shoulder_angle
        
        return angles
