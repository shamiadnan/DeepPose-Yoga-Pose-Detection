"""
DeepPose 2.0 — Session State Machine
======================================
Manages user session state transitions for stable pose detection.
Implements IDLE → TRANSITIONING → HOLDING_POSE state logic.
"""

import os
import sys
import time
from enum import Enum
from typing import Optional, Deque
from collections import deque, Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SessionState(Enum):
    """Possible states for the user's pose session."""
    IDLE = "idle"
    TRANSITIONING = "transitioning"
    HOLDING_POSE = "holding_pose"


@dataclass
class UserSession:
    """Tracks the current state of a user's yoga session.
    
    Attributes:
        state: Current session state
        current_pose: Currently detected pose name (or None)
        consecutive_pose_count: Number of consecutive frames with same pose
        hold_start_time: Timestamp when HOLDING_POSE began
        last_feedback: Last feedback list provided to avoid repetition
        frame_buffer: Rolling window of recent predictions for smoothing
    """
    state: SessionState = SessionState.IDLE
    current_pose: Optional[str] = None
    consecutive_pose_count: int = 0
    hold_start_time: Optional[float] = None
    last_feedback: Optional[list] = None
    frame_buffer: Deque = field(
        default_factory=lambda: deque(maxlen=config.SMOOTHING_WINDOW)
    )


def update_session_state(session: UserSession, pose_name: Optional[str],
                         confidence: float) -> UserSession:
    """Update session state based on current frame's prediction.
    
    Implements the state machine:
    - IDLE → TRANSITIONING: When a pose is detected above confidence threshold
    - TRANSITIONING → HOLDING_POSE: When same pose held for N consecutive frames
    - TRANSITIONING → IDLE: When pose is lost or changes before stabilizing
    - HOLDING_POSE → TRANSITIONING: When the pose changes
    
    Args:
        session: Current UserSession state
        pose_name: Predicted pose name (may be None if no prediction)
        confidence: Prediction confidence [0.0, 1.0]
    
    Returns:
        Updated UserSession object
    
    State transitions follow strictly:
        IDLE → TRANSITIONING (only)
        TRANSITIONING → HOLDING_POSE or IDLE
        HOLDING_POSE → TRANSITIONING (only)
    """
    # Add to frame buffer for smoothing
    if pose_name and confidence >= config.CONFIDENCE_THRESHOLD:
        session.frame_buffer.append(pose_name)
    
    # Below confidence threshold → reset to IDLE
    if confidence < config.CONFIDENCE_THRESHOLD or pose_name is None:
        session.consecutive_pose_count = 0
        session.state = SessionState.IDLE
        session.current_pose = None
        session.hold_start_time = None
        return session
    
    # Check if same pose continues
    if pose_name == session.current_pose:
        session.consecutive_pose_count += 1
    else:
        # Pose changed
        session.current_pose = pose_name
        session.consecutive_pose_count = 1
        session.state = SessionState.TRANSITIONING
        session.hold_start_time = None
    
    # Check if held long enough
    if session.consecutive_pose_count >= config.HOLD_FRAMES_REQUIRED:
        session.state = SessionState.HOLDING_POSE
        if session.hold_start_time is None:
            session.hold_start_time = time.time()
    elif session.consecutive_pose_count > 0:
        session.state = SessionState.TRANSITIONING
    
    return session


def get_smoothed_prediction(session: UserSession) -> Optional[str]:
    """Get the majority-vote pose from the frame buffer.
    
    Uses the rolling window of recent predictions to determine the
    most stable (most frequent) pose, reducing flickering.
    
    Args:
        session: Current UserSession with frame_buffer
    
    Returns:
        Most common pose name in the buffer, or None if buffer is empty
    """
    if not session.frame_buffer:
        return None
    
    counter = Counter(session.frame_buffer)
    most_common = counter.most_common(1)[0][0]
    return most_common
