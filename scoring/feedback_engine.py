"""
DeepPose 2.0 — Feedback Engine
================================
Generates human-readable corrective instructions based on posture
deviations. Uses rule-based lookup for anatomically appropriate feedback.
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ========================
# Correction Rules Database
# ========================
# Maps (joint, direction) to human-readable corrective instruction.
# direction: "more" = angle too small (needs to increase), "less" = angle too large
CORRECTION_RULES: Dict[str, Dict[str, str]] = {
    "left_knee": {
        "more": "Bend your left knee more",
        "less": "Straighten your left knee",
    },
    "right_knee": {
        "more": "Bend your right knee more",
        "less": "Straighten your right knee",
    },
    "left_hip": {
        "more": "Hinge deeper at your left hip",
        "less": "Open your left hip angle",
    },
    "right_hip": {
        "more": "Hinge deeper at your right hip",
        "less": "Open your right hip angle",
    },
    "left_elbow": {
        "more": "Bend your left elbow more",
        "less": "Extend your left arm further",
    },
    "right_elbow": {
        "more": "Bend your right elbow more",
        "less": "Extend your right arm further",
    },
    "shoulder_alignment": {
        "more": "Level your shoulders — they're tilted",
        "less": "Level your shoulders — they're tilted",
    },
}

# Pose-specific overrides for more contextual feedback
POSE_SPECIFIC_RULES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Warrior2_Pose": {
        "left_knee": {
            "more": "Sink deeper — bend your front knee to 90 degrees",
            "less": "Don't over-bend your front knee past your ankle",
        },
        "shoulder_alignment": {
            "more": "Keep your shoulders level and arms parallel to the ground",
            "less": "Keep your shoulders level and arms parallel to the ground",
        },
    },
    "Warrior1_Pose": {
        "left_knee": {
            "more": "Deepen your front knee bend toward 90 degrees",
            "less": "Ease back on the front knee bend",
        },
    },
    "Tree_Pose": {
        "right_knee": {
            "more": "Open your bent knee outward more",
            "less": "Don't force your knee — find a comfortable position",
        },
        "shoulder_alignment": {
            "more": "Keep your hips and shoulders level",
            "less": "Keep your hips and shoulders level",
        },
    },
    "Downdog_Pose": {
        "left_hip": {
            "more": "Push your hips higher toward the ceiling",
            "less": "Ease your hips down slightly",
        },
        "right_hip": {
            "more": "Push your hips higher toward the ceiling",
            "less": "Ease your hips down slightly",
        },
    },
    "Plank_Pose": {
        "left_hip": {
            "more": "Raise your hips — don't let them sag",
            "less": "Lower your hips — don't pike up",
        },
        "right_hip": {
            "more": "Raise your hips — don't let them sag",
            "less": "Lower your hips — don't pike up",
        },
    },
    "Cobra_Pose": {
        "left_elbow": {
            "more": "Bend your elbows slightly to protect your lower back",
            "less": "Press up through your arms for more lift",
        },
        "right_elbow": {
            "more": "Bend your elbows slightly to protect your lower back",
            "less": "Press up through your arms for more lift",
        },
    },
    "Chair_Pose": {
        "left_knee": {
            "more": "Sit deeper — imagine sitting into a chair",
            "less": "Don't over-bend — keep weight in your heels",
        },
        "right_knee": {
            "more": "Sit deeper — imagine sitting into a chair",
            "less": "Don't over-bend — keep weight in your heels",
        },
    },
    "Bridge_Pose": {
        "left_hip": {
            "more": "Lift your hips higher toward the ceiling",
            "less": "Don't overextend — protect your lower back",
        },
        "right_hip": {
            "more": "Lift your hips higher toward the ceiling",
            "less": "Don't overextend — protect your lower back",
        },
    },
}


class FeedbackEngine:
    """Generates corrective feedback from posture deviations.
    
    Uses a rule-based system to convert angular deviations into
    human-readable, actionable instructions. Feedback is filtered
    by a deviation threshold and ordered by severity.
    
    Args:
        deviation_threshold: Minimum deviation (degrees) to trigger feedback
                            (default: 15.0)
    
    Example:
        >>> engine = FeedbackEngine(deviation_threshold=15.0)
        >>> feedback = engine.generate_feedback("Warrior2_Pose", deviations)
        >>> for instruction in feedback:
        ...     print(instruction)
    """
    
    def __init__(self, deviation_threshold: float = None):
        self.deviation_threshold = deviation_threshold or config.DEVIATION_THRESHOLD
    
    def generate_feedback(self, pose_name: str,
                          deviations: Dict[str, float],
                          measured_angles: Dict[str, float] = None,
                          ideal_angles: Dict[str, float] = None) -> List[str]:
        """Generate corrective instructions for deviations above threshold.
        
        Args:
            pose_name: Name of the current yoga pose
            deviations: Dictionary of per-joint deviations in degrees
            measured_angles: Actual measured angles (for direction detection)
            ideal_angles: Ideal target angles (for direction detection)
        
        Returns:
            List of human-readable corrective strings, ordered by severity
            (largest deviation first). Empty list if all deviations are
            below the threshold.
        """
        # Filter joints exceeding threshold
        significant = {
            joint: dev for joint, dev in deviations.items()
            if dev > self.deviation_threshold
        }
        
        if not significant:
            return []
        
        # Sort by severity (largest deviation first)
        sorted_joints = sorted(
            significant.items(), key=lambda x: x[1], reverse=True
        )
        
        feedback = []
        for joint, deviation in sorted_joints:
            instruction = self._get_instruction(
                pose_name, joint, deviation, measured_angles, ideal_angles
            )
            if instruction:
                feedback.append(instruction)
        
        return feedback
    
    def _get_instruction(self, pose_name: str, joint: str, 
                         deviation: float,
                         measured_angles: Dict[str, float] = None,
                         ideal_angles: Dict[str, float] = None) -> str:
        """Get the appropriate corrective instruction for a joint.
        
        Checks pose-specific rules first, then falls back to generic rules.
        
        Args:
            pose_name: Current pose name
            joint: Joint name with deviation
            deviation: Deviation magnitude in degrees
            measured_angles: Actual angles (for direction)
            ideal_angles: Ideal angles (for direction)
        
        Returns:
            Human-readable corrective instruction string
        """
        # Determine direction: does the user need more or less angle?
        direction = "more"  # default
        if measured_angles and ideal_angles and joint in measured_angles and joint in ideal_angles:
            if measured_angles[joint] > ideal_angles[joint]:
                direction = "less"
            else:
                direction = "more"
        
        # Check pose-specific rules first
        if pose_name in POSE_SPECIFIC_RULES:
            pose_rules = POSE_SPECIFIC_RULES[pose_name]
            if joint in pose_rules and direction in pose_rules[joint]:
                return pose_rules[joint][direction]
        
        # Fall back to generic rules
        if joint in CORRECTION_RULES and direction in CORRECTION_RULES[joint]:
            return CORRECTION_RULES[joint][direction]
        
        # Ultimate fallback
        return f"Adjust your {joint.replace('_', ' ')}"
