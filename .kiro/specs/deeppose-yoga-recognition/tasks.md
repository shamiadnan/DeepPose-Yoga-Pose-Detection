# Implementation Plan: DeepPose 2.0 — AI-Powered Yoga Pose Recognition & Correction

## Overview

This implementation plan breaks the DeepPose 2.0 system into discrete, incremental coding tasks organized across 5 phases. Each task builds upon prior work, ensuring no orphaned or disconnected code. The system is implemented in Python using PyTorch, MediaPipe, OpenCV, Streamlit, and pyttsx3. Tasks reference specific requirements and correctness properties from the design document.

## Tasks

- [ ] 1. Update Configuration and Project Structure
  - [ ] 1.1 Update config.py with 11 yoga classes and new project settings
    - Replace the existing 5-class CLASSES list with all 11 yoga classes: Bridge, Chair, Child, Cobra, Downdog, Goddess, Plank, Tree, Triangle, Warrior1, Warrior2
    - Add training configuration constants: IMG_SIZE=224, BATCH_SIZE=32, LEARNING_RATE=0.001, EPOCHS=50, PATIENCE=5, TEST_SIZE=0.2, VAL_SIZE=0.1, RANDOM_STATE=42, NUM_CLASSES=11
    - Update CONFIDENCE_THRESHOLD to 0.5 and SMOOTHING_WINDOW to 10
    - Add HOLD_FRAMES_REQUIRED=10, DEVIATION_THRESHOLD=15.0
    - Add paths: DATASET_DIR, CHECKPOINT_DIR, OUTPUT_DIR, and ensure directories are created if missing
    - Keep MEAN and STD (ImageNet normalization values)
    - _Requirements: 1.1, 4.3, 10.3_

  - [ ] 1.2 Create project directory structure
    - Create directories: models/, checkpoints/, scoring/, voice/, mediapipe_utils/, streamlit_app/, outputs/
    - Create empty `__init__.py` files in models/, scoring/, voice/, mediapipe_utils/ to make them proper Python packages
    - _Requirements: All (project scaffolding)_

- [ ] 2. Implement Dataset Manager and Training Pipeline
  - [ ] 2.1 Implement DatasetManager class in models/train.py
    - Create class DatasetManager with __init__(self, dataset_path, img_size=224, batch_size=32)
    - Implement dataset discovery: recursively walk dataset_path to find all image files organized in subdirectories named by class label
    - Implement load_and_split(test_size=0.2, val_size=0.1, random_state=42) method using sklearn.model_selection.train_test_split with stratify parameter
    - Implement train-time augmentation transforms: Resize(224,224), RandomHorizontalFlip(), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2), ToTensor(), Normalize(ImageNet mean/std)
    - Implement eval-time transforms: Resize(224,224), ToTensor(), Normalize(ImageNet mean/std)
    - Implement get_class_names() returning sorted list of 11 class names
    - Implement get_class_weights() computing inverse-frequency weights from training split as torch.Tensor
    - Raise descriptive ValueError if any class directory is missing or empty
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.2 Write property tests for DatasetManager
    - **Property 2: Stratified Split Preserves Class Distribution** — For any dataset with at least 2 samples per class, verify class proportions in each split match original within 5 percentage points
    - **Property 3: Split Reproducibility** — Splitting twice with same random_state produces identical assignments
    - **Property 4: Class Weight Inverse Proportionality** — Product of class weight and sample count is approximately equal across all classes
    - Use Hypothesis library with custom strategies for generating mock dataset structures
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [ ] 2.3 Implement YogaTrainer class in models/train.py
    - Create class YogaTrainer with __init__(self, num_classes=11, learning_rate=0.001, patience=5, device="cuda")
    - Implement build_model(): load ResNet18 pretrained on ImageNet, freeze all backbone parameters (requires_grad=False), replace model.fc with nn.Linear(512, num_classes)
    - Implement train(train_loader, val_loader, epochs=50): full training loop with CrossEntropyLoss(weight=class_weights), Adam optimizer on model.fc.parameters() only
    - Track train_loss, val_loss, train_acc, val_acc per epoch in history dict
    - Implement early stopping: if val_loss doesn't improve for `patience` consecutive epochs, stop and retain best weights
    - Implement save_checkpoint(path="checkpoints/best_model.pth"): save model state_dict
    - Auto-save best_model.pth whenever validation loss improves
    - Return training history dictionary on completion
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 2.4 Write property test for early stopping
    - **Property 5: Early Stopping Correctness** — For any sequence of validation losses where the last N consecutive values (N=patience) do not improve, training terminates at exactly that epoch
    - Use Hypothesis to generate synthetic loss sequences and verify termination behavior
    - **Validates: Requirement 2.4**

- [ ] 3. Implement Model Evaluation
  - [ ] 3.1 Create models/evaluate.py with evaluation metrics
    - Implement evaluate_model(model, test_loader, class_names, device) function
    - Compute overall accuracy, per-class precision, recall, and F1-score using sklearn.metrics.classification_report
    - Generate and save confusion matrix as outputs/confusion_matrix.png using seaborn heatmap
    - Generate and save accuracy curves as outputs/accuracy_curve.png (train_acc vs val_acc per epoch)
    - Generate and save loss curves as outputs/loss_curve.png (train_loss vs val_loss per epoch)
    - Accept training history dict for curve generation
    - Return metrics dictionary with all computed values
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Checkpoint — Verify training pipeline
  - Ensure DatasetManager loads dataset correctly, YogaTrainer completes training with early stopping, evaluation metrics and plots are generated in outputs/. Ask the user if questions arise.

- [ ] 5. Implement MediaPipe Pose Detector
  - [ ] 5.1 Create mediapipe_utils/pose_detector.py
    - Implement PoseDetector class with __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    - Implement detect(frame: np.ndarray) -> Optional[List]: extract 33 landmarks with (x, y, z, visibility) from BGR frame; return None if no pose detected
    - Implement draw_skeleton(frame, landmarks) -> np.ndarray: draw skeleton connections using mp_drawing utilities; return annotated frame copy
    - Implement extract_feature_vector(landmarks) -> np.ndarray: flatten all 33 landmarks × 4 values into a 132-element numpy array
    - Handle edge cases: return None for frames with no detectable person
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 15.1_

  - [ ]* 5.2 Write property test for feature vector dimensionality
    - **Property 15: Feature Vector Dimensionality** — For any valid 33 landmarks, extract_feature_vector() returns exactly 132 elements
    - Use Hypothesis to generate random landmark arrays and verify output shape
    - **Validates: Requirements 5.4, 15.1**

- [ ] 6. Implement Inference Engine
  - [ ] 6.1 Create models/inference.py
    - Implement PoseInferenceEngine class with __init__(self, model_path="checkpoints/best_model.pth", class_names=None, device="cpu")
    - Load model checkpoint on initialization; raise FileNotFoundError with descriptive message if checkpoint missing
    - Implement predict(frame: np.ndarray) -> Tuple[str, float]: preprocess BGR frame (resize 224×224, normalize), run forward pass, return (pose_name, confidence) from softmax
    - Implement predict_batch(frames: List[np.ndarray]) -> List[Tuple[str, float]]: batch inference for multiple frames
    - Ensure model is set to eval mode with torch.no_grad() context
    - Implement automatic GPU→CPU fallback if CUDA OOM occurs
    - _Requirements: 4.1, 4.2, 4.4, 17.2, 17.6_

  - [ ]* 6.2 Write property test for classification completeness
    - **Property 1: Classification Completeness** — For any valid BGR frame, predict() returns one of exactly 11 class names and confidence in [0.0, 1.0]
    - Use Hypothesis to generate random BGR arrays of various sizes
    - **Validates: Requirements 4.1, 4.2**

- [ ] 7. Checkpoint — Verify real-time inference pipeline
  - Ensure PoseDetector extracts landmarks, InferenceEngine classifies poses, and both work on test images. Ask the user if questions arise.

- [ ] 8. Implement Angle Calculator
  - [ ] 8.1 Create scoring/angle_utils.py
    - Implement AngleCalculator class
    - Implement calculate_angle(point_a, point_b, point_c) -> float: compute angle at point_b using dot product formula angle = arccos(dot(BA, BC) / (|BA| * |BC| + 1e-8)); clamp cos_angle to [-1, 1] to avoid numerical errors; return degrees in [0, 180]
    - Implement get_all_angles(landmarks) -> Dict[str, float]: compute 7 joint angles (left_knee, right_knee, left_hip, right_hip, left_elbow, right_elbow, shoulder_alignment) using appropriate MediaPipe landmark triplets
    - Handle degenerate case (coincident points) by using epsilon=1e-8 in denominator
    - Handle low-visibility landmarks (visibility < 0.3): exclude affected joints from output dictionary
    - Ensure total computation time < 1ms using numpy vectorized operations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 17.4_

  - [ ]* 8.2 Write property tests for AngleCalculator
    - **Property 6: Angle Validity and Bounds** — For any three non-coincident points, calculate_angle returns value in [0.0, 180.0] with no NaN/Inf
    - **Property 7: Angle Output Completeness** — For any valid 33 landmarks with visibility > 0.3, get_all_angles returns dict with exactly 7 keys, all values in [0.0, 180.0]
    - **Property 17: Low Visibility Joint Exclusion** — For landmarks with visibility < 0.3, affected joints are excluded from output
    - Use Hypothesis with numpy array strategies
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.5, 17.4**

- [ ] 9. Implement Posture Scorer
  - [ ] 9.1 Create scoring/posture_score.py
    - Implement ideal angle reference database (IDEAL_ANGLES dict) with target angles for all 7 joints across all 11 yoga poses
    - Implement PostureScorer class with __init__(self, ideal_angles_db=None) defaulting to built-in IDEAL_ANGLES
    - Implement score(pose_name, measured_angles) -> Tuple[float, Dict[str, float]]:
      - Compute per-joint deviation = abs(measured - ideal)
      - Apply joint-specific weights via get_joint_weights(pose_name)
      - Compute normalized_penalty = min(deviation / 45.0, 1.0) per joint
      - Compute overall_score = max(0.0, (1.0 - weighted_penalty / total_weight) * 100.0)
      - Return (overall_score, deviations_dict)
    - Raise ValueError with descriptive message if pose_name not in database
    - Handle partial scoring when some joints are excluded (adjust total_weight)
    - Ensure score of 100.0 when all measured angles exactly match ideal
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 17.4_

  - [ ]* 9.2 Write property tests for PostureScorer
    - **Property 8: Score Bounds** — For any valid pose and measured angles in [0, 180], score is in [0.0, 100.0]
    - **Property 9: Score Monotonicity** — Decreasing one joint's deviation while keeping others constant produces score >= original
    - **Property 10: Posture Deviation Correctness** — Per-joint deviations equal absolute difference between measured and ideal
    - Use Hypothesis with dictionaries of float strategies
    - **Validates: Requirements 7.2, 7.3, 7.5**

- [ ] 10. Implement Feedback Engine
  - [ ] 10.1 Create scoring/feedback_engine.py
    - Implement FeedbackEngine class with __init__(self, deviation_threshold=15.0)
    - Implement generate_feedback(pose_name, deviations) -> List[str]:
      - Filter joints where deviation > deviation_threshold (15 degrees)
      - Sort filtered joints by deviation magnitude descending (largest first)
      - Map each joint + deviation to human-readable corrective instruction using pose-specific rule lookup
      - Return list of actionable strings (e.g., "Bend your left knee more" not "left_knee deviation: 25")
      - Return empty list if all deviations below threshold
    - Implement pose-specific correction rules dictionary mapping (pose, joint, direction) to anatomically appropriate instructions
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 10.2 Write property tests for FeedbackEngine
    - **Property 11: Feedback Threshold Filtering** — Feedback only for joints with deviation > 15 degrees; empty list when all below threshold
    - **Property 12: Feedback Severity Ordering** — Output list is ordered by deviation magnitude descending
    - Use Hypothesis to generate random deviation dictionaries
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [ ] 11. Checkpoint — Verify biomechanical analysis pipeline
  - Ensure AngleCalculator computes valid angles from landmarks, PostureScorer returns scores in [0, 100], and FeedbackEngine generates ordered corrective text. Ask the user if questions arise.

- [ ] 12. Implement Voice Feedback
  - [ ] 12.1 Create voice/voice_feedback.py
    - Implement VoiceFeedback class with __init__(self, rate=150)
    - Initialize pyttsx3 engine in a background daemon thread
    - Implement speak(text: str) -> None: queue speech in background thread; return immediately (non-blocking); skip if text equals last spoken text (deduplication)
    - Implement stop() -> None: terminate ongoing speech, clean up engine resources gracefully
    - Handle pyttsx3 initialization failure: catch exceptions, log warning, set self.available=False; system continues with visual-only feedback
    - Implement is_available() -> bool property for status checking
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 12.2 Write property test for voice deduplication
    - **Property 14: Voice Deduplication** — Speech triggered only when current text differs from last spoken text; consecutive identical texts do not trigger additional speech
    - Use Hypothesis to generate sequences of text inputs and verify deduplication behavior
    - **Validates: Requirements 9.2, 9.3**

- [ ] 13. Implement Session State Machine
  - [ ] 13.1 Create session state machine in models/inference.py or a dedicated session module
    - Implement SessionState enum: IDLE, TRANSITIONING, HOLDING_POSE
    - Implement UserSession dataclass with: state, current_pose, consecutive_pose_count, hold_start_time, last_feedback, frame_buffer (deque with maxlen=SMOOTHING_WINDOW)
    - Implement update_session_state(session, pose_name, confidence) -> UserSession:
      - If confidence < 0.5: reset to IDLE, clear pose count
      - If pose matches current: increment consecutive count
      - If pose differs: reset count to 1, set state to TRANSITIONING
      - If consecutive count >= HOLD_FRAMES_REQUIRED (10): transition to HOLDING_POSE
      - Only permit valid transitions per state diagram: IDLE→TRANSITIONING, TRANSITIONING→HOLDING_POSE/IDLE, HOLDING_POSE→TRANSITIONING
    - Implement majority vote smoothing using frame_buffer for stable predictions
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 4.5_

  - [ ]* 13.2 Write property tests for Session State Machine
    - **Property 13: State Machine Valid Transitions** — For any sequence of inputs, only valid state transitions occur per the defined state diagram
    - **Property 16: Rolling Window Majority Vote** — The smoothing function returns the mode of the buffer
    - Use Hypothesis to generate sequences of (pose_name, confidence) tuples
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.5, 10.6, 4.5**

- [ ] 14. Checkpoint — Verify voice and state machine integration
  - Ensure VoiceFeedback speaks non-blocking with deduplication, state machine transitions correctly through IDLE→TRANSITIONING→HOLDING_POSE. Ask the user if questions arise.

- [ ] 15. Implement Streamlit Dashboard — Home and Image Upload
  - [ ] 15.1 Create streamlit_app/app.py with multi-page navigation and Home page
    - Set up Streamlit multi-page app with sidebar navigation: Home, Image Upload, Live Camera, Model Performance
    - Implement Home page: display project name (DeepPose 2.0), description, list of 11 supported poses, usage instructions
    - Configure Streamlit page settings (title, layout, icon)
    - _Requirements: 11.1, 11.2_

  - [ ] 15.2 Implement Image Upload page in streamlit_app/app.py
    - Add file uploader accepting image formats (jpg, jpeg, png)
    - Validate uploaded file size (reject > 10MB) and format before processing
    - On upload: run InferenceEngine.predict() to get pose name and confidence
    - Run PoseDetector.detect() to get landmarks, draw skeleton overlay
    - Compute joint angles via AngleCalculator, score via PostureScorer
    - Generate feedback via FeedbackEngine
    - Display results: pose name, confidence %, posture score (0-100), skeleton overlay image, ordered feedback list
    - Handle case where no person detected: display "No pose detected" message
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 18.2_

  - [ ]* 15.3 Write property test for upload validation
    - **Property 18: Upload Validation** — Files exceeding 10MB or with invalid format are rejected before processing
    - Use Hypothesis to generate file-like objects of various sizes
    - **Validates: Requirement 18.2**

- [ ] 16. Implement Streamlit Dashboard — Live Camera and Performance
  - [ ] 16.1 Implement Live Camera page in streamlit_app/app.py
    - Capture webcam frames using OpenCV via Streamlit's camera input or custom loop
    - Run full pipeline per frame: classification → landmark detection → angle calculation → scoring → feedback
    - Integrate session state machine: only activate scoring when in HOLDING_POSE state
    - Display overlays: pose name, confidence score, posture score, skeleton, feedback list, voice status
    - Integrate VoiceFeedback: trigger non-blocking speech for top correction when feedback changes
    - Handle webcam access failure: display error message, suggest Image Upload as alternative
    - Maintain per-session independent state using st.session_state
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 16.1, 17.1, 17.3, 18.1, 18.3_

  - [ ] 16.2 Implement Model Performance page in streamlit_app/app.py
    - Load and display overall accuracy, per-class precision, recall, and F1-score metrics
    - Display confusion_matrix.png, accuracy_curve.png, loss_curve.png from outputs/ directory
    - Handle missing files gracefully: show "Model has not been trained yet" message if files not found
    - _Requirements: 14.1, 14.2, 14.3_

- [ ] 17. Checkpoint — Verify Streamlit dashboard
  - Ensure all 4 Streamlit pages render correctly: Home shows info, Image Upload processes images with full pipeline, Live Camera integrates real-time detection, Model Performance displays metrics. Ask the user if questions arise.

- [ ] 18. Create requirements.txt and update README.md
  - [ ] 18.1 Create requirements.txt with all project dependencies
    - Include: torch>=2.0, torchvision>=0.15, numpy>=1.24, opencv-python>=4.8, mediapipe>=0.10, scikit-learn>=1.3, streamlit>=1.28, pyttsx3>=2.90, matplotlib>=3.7, seaborn>=0.12, Pillow>=9.0, hypothesis>=6.0 (dev dependency)
    - Pin major versions for reproducibility
    - _Requirements: All (dependency management)_

  - [ ] 18.2 Update README.md with project documentation
    - Add project title, description, features overview
    - Document all 11 supported yoga poses
    - Add installation instructions (pip install -r requirements.txt)
    - Add usage instructions: how to train, how to run Streamlit app
    - Document folder structure and module descriptions
    - Add performance requirements and system architecture overview
    - _Requirements: All (documentation)_

- [ ] 19. Integration and End-to-End Wiring
  - [ ] 19.1 Wire complete real-time pipeline end-to-end
    - Ensure all modules connect correctly: frame → InferenceEngine + PoseDetector → AngleCalculator → PostureScorer → FeedbackEngine → VoiceFeedback
    - Verify state machine gates scoring (only score in HOLDING_POSE)
    - Verify privacy: no webcam frames saved to disk (in-memory only)
    - Test GPU→CPU fallback path
    - Validate full pipeline latency < 33ms per frame target
    - _Requirements: 4.5, 10.4, 16.1, 17.1, 17.5, 17.6, 18.1_

  - [ ]* 19.2 Write integration tests for end-to-end pipeline
    - Test complete flow: test image → predict → landmarks → angles → score → feedback
    - Verify all modules integrate without errors on valid inputs
    - Verify graceful degradation: no landmarks → skip scoring, no model → FileNotFoundError, no camera → error message
    - **Validates: Requirements 16.1, 17.1, 17.2, 17.3, 17.4, 17.5**

- [ ] 20. Final Checkpoint — Complete system validation
  - Ensure all tests pass, all modules integrate correctly, Streamlit dashboard runs with all 4 pages functional, voice feedback works non-blocking, state machine manages sessions. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at natural breakpoints
- Property tests use the Hypothesis library (Python) and validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python throughout, using PyTorch, MediaPipe, OpenCV, Streamlit, and pyttsx3
- All 18 correctness properties from the design are covered by testing sub-tasks
- The alternative tabular classification path (Requirement 15) is partially addressed via feature vector extraction in task 5.1; full XGBoost/RF training can be added as a follow-up

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "5.1", "8.1", "12.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "5.2", "8.2", "12.2"] },
    { "id": 3, "tasks": ["2.4", "3.1", "6.1", "9.1"] },
    { "id": 4, "tasks": ["6.2", "9.2", "10.1", "13.1"] },
    { "id": 5, "tasks": ["10.2", "13.2", "15.1"] },
    { "id": 6, "tasks": ["15.2", "15.3", "16.2"] },
    { "id": 7, "tasks": ["16.1", "18.1", "18.2"] },
    { "id": 8, "tasks": ["19.1"] },
    { "id": 9, "tasks": ["19.2"] }
  ]
}
```
