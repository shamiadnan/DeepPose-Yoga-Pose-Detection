"""
DeepPose 2.0 — Posture Scorer
===============================
Scores posture quality (0-100) based on angular deviation from ideal
reference angles for each of the 11 yoga poses.
"""

import os
import sys
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ========================
# Ideal Angle Database
# ========================
# Target angles (in degrees) for each pose and joint.
# These represent biomechanically correct form for each yoga pose.
IDEAL_ANGLES: Dict[str, Dict[str, float]] = {
    "Bridge_Pose": {
        "left_knee": 90.0,
        "right_knee": 90.0,
        "left_hip": 160.0,
        "right_hip": 160.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Chair_Pose": {
        "left_knee": 100.0,
        "right_knee": 100.0,
        "left_hip": 100.0,
        "right_hip": 100.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Child_Pose": {
        "left_knee": 40.0,
        "right_knee": 40.0,
        "left_hip": 50.0,
        "right_hip": 50.0,
        "left_elbow": 160.0,
        "right_elbow": 160.0,
        "shoulder_alignment": 0.0,
    },
    "Cobra_Pose": {
        "left_knee": 180.0,
        "right_knee": 180.0,
        "left_hip": 160.0,
        "right_hip": 160.0,
        "left_elbow": 140.0,
        "right_elbow": 140.0,
        "shoulder_alignment": 0.0,
    },
    "Downdog_Pose": {
        "left_knee": 180.0,
        "right_knee": 180.0,
        "left_hip": 70.0,
        "right_hip": 70.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Goddess_Pose": {
        "left_knee": 90.0,
        "right_knee": 90.0,
        "left_hip": 90.0,
        "right_hip": 90.0,
        "left_elbow": 90.0,
        "right_elbow": 90.0,
        "shoulder_alignment": 0.0,
    },
    "Plank_Pose": {
        "left_knee": 180.0,
        "right_knee": 180.0,
        "left_hip": 180.0,
        "right_hip": 180.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Tree_Pose": {
        "left_knee": 180.0,
        "right_knee": 90.0,
        "left_hip": 180.0,
        "right_hip": 45.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Triangle_Pose": {
        "left_knee": 180.0,
        "right_knee": 180.0,
        "left_hip": 90.0,
        "right_hip": 90.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 5.0,
    },
    "Warrior1_Pose": {
        "left_knee": 90.0,
        "right_knee": 180.0,
        "left_hip": 100.0,
        "right_hip": 160.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
    "Warrior2_Pose": {
        "left_knee": 90.0,
        "right_knee": 180.0,
        "left_hip": 90.0,
        "right_hip": 160.0,
        "left_elbow": 180.0,
        "right_elbow": 180.0,
        "shoulder_alignment": 0.0,
    },
}

# ========================
# Joint Weights per Pose
# ========================
# Higher weight = more important for correct form in that specific pose
JOINT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Bridge_Pose": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 0.5, "right_elbow": 0.5,
        "shoulder_alignment": 1.5,
    },
    "Chair_Pose": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "shoulder_alignment": 1.0,
    },
    "Child_Pose": {
        "left_knee": 1.5, "right_knee": 1.5,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "shoulder_alignment": 1.0,
    },
    "Cobra_Pose": {
        "left_knee": 1.0, "right_knee": 1.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 2.0, "right_elbow": 2.0,
        "shoulder_alignment": 1.5,
    },
    "Downdog_Pose": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "shoulder_alignment": 1.0,
    },
    "Goddess_Pose": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "shoulder_alignment": 1.0,
    },
    "Plank_Pose": {
        "left_knee": 1.5, "right_knee": 1.5,
        "left_hip": 2.5, "right_hip": 2.5,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "shoulder_alignment": 1.5,
    },
    "Tree_Pose": {
        "left_knee": 1.5, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "shoulder_alignment": 1.5,
    },
    "Triangle_Pose": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 2.0, "right_hip": 2.0,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "shoulder_alignment": 1.5,
    },
    "Warrior1_Pose": {
        "left_knee": 2.5, "right_knee": 1.5,
        "left_hip": 2.0, "right_hip": 1.5,
        "left_elbow": 1.0, "right_elbow": 1.0,
        "shoulder_alignment": 1.5,
    },
    "Warrior2_Pose": {
        "left_knee": 2.5, "right_knee": 1.5,
        "left_hip": 2.0, "right_hip": 1.5,
        "left_elbow": 1.5, "right_elbow": 1.5,
        "shoulder_alignment": 1.5,
    },
}


class PostureScorer:
    """Scores posture quality based on angular deviation from ideal poses.
    
    Computes a weighted overall score (0-100) where 100 is perfect alignment
    with the ideal pose angles. Uses per-joint weighting to emphasize the
    most important joints for each specific pose.
    
    Scoring formula:
        penalty_per_joint = min(deviation / MAX_DEVIATION, 1.0) * weight
        overall_score = max(0, (1 - sum(penalties) / sum(weights)) * 100)
    
    Args:
        ideal_angles_db: Dictionary of ideal angles per pose (default: IDEAL_ANGLES)
        joint_weights_db: Dictionary of joint weights per pose (default: JOINT_WEIGHTS)
    
    Example:
        >>> scorer = PostureScorer()
        >>> score, deviations = scorer.score("Warrior2_Pose", measured_angles)
        >>> print(f"Score: {score:.0f}/100")
    """
    
    def __init__(self, ideal_angles_db: Dict[str, Dict[str, float]] = None,
                 joint_weights_db: Dict[str, Dict[str, float]] = None):
        self.ideal_angles = ideal_angles_db or IDEAL_ANGLES
        self.joint_weights = joint_weights_db or JOINT_WEIGHTS
    
    def score(self, pose_name: str, 
              measured_angles: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute posture score and per-joint deviations.
        
        Args:
            pose_name: Name of the detected yoga pose (must be in ideal database)
            measured_angles: Dictionary of measured joint angles in degrees
        
        Returns:
            Tuple of (overall_score, deviations):
            - overall_score: Float in [0.0, 100.0] where 100 = perfect
            - deviations: Dict mapping joint names to absolute deviation in degrees
        
        Raises:
            ValueError: If pose_name is not in the ideal angle database
        """
        if pose_name not in self.ideal_angles:
            raise ValueError(
                f"Unknown pose: '{pose_name}'. "
                f"Available poses: {list(self.ideal_angles.keys())}"
            )
        
        ideal = self.ideal_angles[pose_name]
        weights = self.joint_weights.get(pose_name, {})
        
        deviations = {}
        weighted_penalty = 0.0
        total_weight = 0.0
        
        # Only score joints that are present in both measured and ideal
        for joint in measured_angles:
            if joint not in ideal:
                continue
            
            # Compute absolute deviation
            deviation = abs(measured_angles[joint] - ideal[joint])
            deviations[joint] = deviation
            
            # Get weight for this joint (default: 1.0)
            weight = weights.get(joint, 1.0)
            total_weight += weight
            
            # Normalized penalty: 0 at 0°, 1.0 at MAX_DEVIATION (45°)
            normalized_penalty = min(deviation / config.MAX_DEVIATION, 1.0)
            weighted_penalty += normalized_penalty * weight
        
        # Calculate overall score
        if total_weight > 0:
            penalty_ratio = weighted_penalty / total_weight
            overall_score = max(0.0, (1.0 - penalty_ratio) * 100.0)
        else:
            overall_score = 0.0
        
        return overall_score, deviations
    
    def get_ideal_angles(self, pose_name: str) -> Optional[Dict[str, float]]:
        """Get ideal angles for a specific pose.
        
        Args:
            pose_name: Name of the yoga pose
        
        Returns:
            Dictionary of ideal angles or None if pose not found
        """
        return self.ideal_angles.get(pose_name)
