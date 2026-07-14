# Requirements Document

## Introduction

DeepPose 2.0 is an AI-powered Yoga Pose Recognition and Correction System that detects 11 yoga poses in real time via webcam, scores posture quality using biomechanical angle analysis, and provides corrective voice feedback. The system combines deep learning (ResNet18 transfer learning) for pose classification with MediaPipe for skeleton landmark extraction, delivering results through a Streamlit web dashboard. This document specifies the formal requirements derived from the technical design, organized by development phase.

## Glossary

- **System**: The DeepPose 2.0 yoga pose recognition and correction application
- **Dataset_Manager**: The module responsible for dataset loading, splitting, and augmentation
- **Training_Pipeline**: The module that manages ResNet18 transfer learning with early stopping
- **Inference_Engine**: The module that loads a trained model and performs pose classification on frames
- **Pose_Detector**: The MediaPipe wrapper module for landmark extraction and skeleton visualization
- **Angle_Calculator**: The module that computes joint angles from 3D landmark positions using vector dot product
- **Posture_Scorer**: The module that scores pose quality (0-100) based on angular deviation from ideal reference angles
- **Feedback_Engine**: The module that generates human-readable corrective instructions from score deviations
- **Voice_Feedback**: The pyttsx3-based module providing non-blocking text-to-speech output
- **Dashboard**: The Streamlit multi-page web interface integrating all system components
- **Landmark**: A body point detected by MediaPipe with (x, y, z, visibility) coordinates
- **Joint_Angle**: The angle in degrees at a body joint calculated from three landmark points
- **Ideal_Angle_Database**: The reference database containing target angles for each pose and joint
- **Session_State_Machine**: The state controller managing user session transitions (IDLE, TRANSITIONING, HOLDING_POSE)
- **Frame_Buffer**: A rolling window of recent frame predictions used for stable pose detection
- **Confidence_Threshold**: The minimum softmax probability (0.5) required to accept a pose prediction
- **Deviation_Threshold**: The minimum angular deviation (15 degrees) required to trigger corrective feedback
- **Class_Weights**: Inverse-frequency weights computed from the dataset to handle class imbalance during training

## Requirements

### Requirement 1: Dataset Loading and Preparation

**User Story:** As a machine learning engineer, I want to automatically load and split the yoga pose dataset with stratification, so that training, validation, and test sets preserve class distributions without manual file organization.

#### Acceptance Criteria

1. WHEN a dataset directory path is provided, THE Dataset_Manager SHALL recursively discover all image files organized in subdirectories named by class label for exactly 11 yoga classes (Bridge, Chair, Child, Cobra, Downdog, Goddess, Plank, Tree, Triangle, Warrior1, Warrior2)
2. WHEN the dataset is loaded, THE Dataset_Manager SHALL perform stratified splitting using sklearn train_test_split into train, validation, and test sets with configurable ratios (default: 70% train, 10% validation, 20% test)
3. WHEN splitting the dataset, THE Dataset_Manager SHALL use a fixed random_state (default: 42) for reproducibility
4. WHEN computing class weights, THE Dataset_Manager SHALL calculate inverse-frequency weights from the training split to address class imbalance
5. WHEN creating training DataLoaders, THE Dataset_Manager SHALL apply data augmentation transforms: Resize(224, 224), RandomHorizontalFlip, RandomRotation(15 degrees), ColorJitter(brightness=0.2, contrast=0.2), and Normalize with ImageNet mean and standard deviation
6. WHEN creating validation and test DataLoaders, THE Dataset_Manager SHALL apply only Resize(224, 224) and Normalize with ImageNet mean and standard deviation without random augmentation
7. IF a dataset directory does not contain valid image files for any class, THEN THE Dataset_Manager SHALL raise a descriptive error indicating which classes are missing

### Requirement 2: ResNet18 Training Pipeline

**User Story:** As a machine learning engineer, I want to train a ResNet18 model using transfer learning with early stopping, so that I can achieve high classification accuracy without overfitting on the limited dataset.

#### Acceptance Criteria

1. WHEN building the model, THE Training_Pipeline SHALL load a ResNet18 pretrained on ImageNet, freeze all convolutional backbone parameters, and replace the final fully connected layer with a new Linear layer mapping 512 features to 11 output classes
2. WHEN training begins, THE Training_Pipeline SHALL use CrossEntropyLoss weighted by class weights and Adam optimizer with a configurable learning rate (default: 0.001)
3. WHILE training, THE Training_Pipeline SHALL track training loss, validation loss, training accuracy, and validation accuracy for each epoch
4. WHEN validation loss does not improve for a configurable number of consecutive epochs (patience, default: 5), THE Training_Pipeline SHALL stop training early and retain the best model weights
5. WHEN validation loss improves, THE Training_Pipeline SHALL save the model state dictionary as best_model.pth in the checkpoints directory
6. WHEN training completes, THE Training_Pipeline SHALL output a training history dictionary containing per-epoch metrics for visualization

### Requirement 3: Model Evaluation and Metrics

**User Story:** As a machine learning engineer, I want comprehensive evaluation metrics and visualizations after training, so that I can assess model performance across all 11 yoga classes.

#### Acceptance Criteria

1. WHEN evaluation is triggered on the test set, THE System SHALL compute overall accuracy, per-class precision, per-class recall, and per-class F1 score using the best saved checkpoint
2. WHEN evaluation completes, THE System SHALL generate and save a confusion matrix visualization as confusion_matrix.png in the outputs directory
3. WHEN evaluation completes, THE System SHALL generate and save training/validation accuracy curves as accuracy_curve.png and loss curves as loss_curve.png in the outputs directory
4. WHEN evaluation completes, THE System SHALL produce a complete classification report covering all 11 classes with support counts

### Requirement 4: Real-Time Pose Classification

**User Story:** As a yoga practitioner, I want the system to classify my pose in real time from webcam input, so that I can receive immediate identification of which yoga pose I am performing.

#### Acceptance Criteria

1. WHEN a webcam frame is captured, THE Inference_Engine SHALL preprocess it (resize to 224x224, normalize with ImageNet statistics) and produce a pose prediction with softmax confidence score
2. THE Inference_Engine SHALL return exactly one of the 11 defined yoga class names for any valid input frame
3. WHEN the confidence score is below the Confidence_Threshold (0.5), THE System SHALL label the prediction as uncertain and not activate scoring
4. WHEN predicting a pose, THE Inference_Engine SHALL complete single-frame inference within 50 milliseconds on CPU or within 20 milliseconds on GPU
5. WHEN a rolling window of frames (default: 10 frames) contains a majority prediction for one pose, THE System SHALL use that majority pose as the stable prediction to prevent flickering

### Requirement 5: MediaPipe Landmark Detection and Skeleton Visualization

**User Story:** As a yoga practitioner, I want to see my body skeleton overlaid on the video feed, so that I can visually confirm the system is tracking my body correctly.

#### Acceptance Criteria

1. WHEN a frame is provided, THE Pose_Detector SHALL extract 33 pose landmarks each with (x, y, z, visibility) values using MediaPipe Pose
2. WHEN landmarks are detected, THE Pose_Detector SHALL draw skeleton connections on the frame using MediaPipe's drawing utilities
3. IF no landmarks are detected in a frame, THEN THE Pose_Detector SHALL return None and the system shall skip the scoring pipeline for that frame
4. WHEN extracting a feature vector for the alternative classification path, THE Pose_Detector SHALL produce a flat numpy array of exactly 132 values (33 landmarks multiplied by 4 values each)

### Requirement 6: Joint Angle Calculation

**User Story:** As a biomechanics module developer, I want to calculate joint angles from landmark positions using vector mathematics, so that the system can quantify how well a user's pose matches the ideal form.

#### Acceptance Criteria

1. WHEN three landmark points (A, B, C) are provided, THE Angle_Calculator SHALL compute the angle at point B using the vector dot product formula: angle = arccos(dot(BA, BC) / (|BA| * |BC|))
2. THE Angle_Calculator SHALL return angle values clamped to the range [0.0, 180.0] degrees for any valid input
3. WHEN landmarks are available, THE Angle_Calculator SHALL compute all 7 joint angles: left knee, right knee, left hip, right hip, left elbow, right elbow, and shoulder alignment
4. THE Angle_Calculator SHALL complete computation of all 7 joint angles within 1 millisecond
5. IF any two input points are coincident (zero-length vector), THEN THE Angle_Calculator SHALL handle the degenerate case by adding a small epsilon (1e-8) to the denominator to avoid division by zero

### Requirement 7: Posture Scoring

**User Story:** As a yoga practitioner, I want a numerical score (0-100) representing how close my pose is to the ideal form, so that I can track my improvement over time.

#### Acceptance Criteria

1. THE Posture_Scorer SHALL maintain an ideal angle reference database containing target angles for all 7 joints across all 11 yoga poses
2. WHEN a pose name and measured angles are provided, THE Posture_Scorer SHALL compute per-joint deviation as the absolute difference between measured and ideal angles
3. WHEN computing the overall score, THE Posture_Scorer SHALL apply joint-specific weights and calculate a weighted penalty normalized to produce a score in the range [0.0, 100.0]
4. WHEN all measured angles exactly match the ideal angles, THE Posture_Scorer SHALL return a score of 100.0
5. WHEN any joint's deviation decreases while all other deviations remain constant, THE Posture_Scorer SHALL return a score that is greater than or equal to the previous score (monotonicity)
6. IF the pose name does not exist in the ideal angle database, THEN THE Posture_Scorer SHALL raise a descriptive error

### Requirement 8: Corrective Feedback Generation

**User Story:** As a yoga practitioner, I want human-readable corrective instructions for my pose, so that I can understand exactly what body adjustments to make.

#### Acceptance Criteria

1. WHEN deviations are provided, THE Feedback_Engine SHALL generate corrective text only for joints where the deviation exceeds the Deviation_Threshold (15 degrees)
2. WHEN multiple joints exceed the threshold, THE Feedback_Engine SHALL order the feedback list by severity with the largest deviation first
3. WHEN no joints exceed the deviation threshold, THE Feedback_Engine SHALL return an empty feedback list
4. THE Feedback_Engine SHALL produce actionable, human-readable instructions (e.g., "Bend your left knee more" rather than "left_knee deviation: 25 degrees")
5. WHEN generating feedback for a specific pose, THE Feedback_Engine SHALL use pose-specific correction rules that map joint deviations to anatomically appropriate instructions

### Requirement 9: Non-Blocking Voice Feedback

**User Story:** As a yoga practitioner, I want spoken corrective feedback that does not freeze the video feed, so that I can continue practicing while hearing guidance.

#### Acceptance Criteria

1. WHEN the speak function is called with text, THE Voice_Feedback SHALL queue speech in a background thread and return immediately without blocking the caller
2. WHEN identical feedback text is provided consecutively, THE Voice_Feedback SHALL not repeat the speech (deduplication)
3. WHEN new feedback text differs from the last spoken text, THE Voice_Feedback SHALL speak the new feedback
4. WHEN the stop function is called, THE Voice_Feedback SHALL terminate any ongoing speech and clean up resources gracefully
5. IF the pyttsx3 engine fails to initialize, THEN THE Voice_Feedback SHALL log a warning and the system shall continue operating with visual-only feedback

### Requirement 10: Session State Machine

**User Story:** As a yoga practitioner, I want the system to recognize when I am stably holding a pose versus transitioning, so that feedback is only provided when I am committed to a pose.

#### Acceptance Criteria

1. WHEN the system starts or confidence drops below the Confidence_Threshold, THE Session_State_Machine SHALL transition to the IDLE state
2. WHEN a pose is detected with confidence above the threshold but has not been held for the required number of frames, THE Session_State_Machine SHALL be in the TRANSITIONING state
3. WHEN the same pose is detected consecutively for the required number of frames (default: 10 frames), THE Session_State_Machine SHALL transition to the HOLDING_POSE state
4. WHILE in HOLDING_POSE state, THE System SHALL activate the scoring and feedback pipeline
5. WHEN the detected pose changes while in HOLDING_POSE state, THE Session_State_Machine SHALL transition back to TRANSITIONING
6. THE Session_State_Machine SHALL only permit transitions following the defined state diagram: IDLE to TRANSITIONING, TRANSITIONING to HOLDING_POSE or IDLE, HOLDING_POSE to TRANSITIONING

### Requirement 11: Streamlit Dashboard — Home Page

**User Story:** As a user, I want a welcoming home page that explains the project and how to use it, so that I can understand the system before starting.

#### Acceptance Criteria

1. WHEN the user navigates to the Home page, THE Dashboard SHALL display project information including the system name, description, supported poses, and usage instructions
2. WHEN the home page loads, THE Dashboard SHALL render a navigation sidebar allowing access to all 4 pages (Home, Image Upload, Live Camera, Model Performance)

### Requirement 12: Streamlit Dashboard — Image Upload Page

**User Story:** As a yoga practitioner, I want to upload a yoga pose image and receive pose classification, posture score, and corrective feedback, so that I can analyze my form from photographs.

#### Acceptance Criteria

1. WHEN a user uploads an image file, THE Dashboard SHALL preprocess it and run the Inference_Engine to predict the yoga pose with confidence score
2. WHEN pose prediction succeeds, THE Dashboard SHALL detect landmarks, calculate joint angles, compute the posture score, and display corrective feedback
3. WHEN displaying results, THE Dashboard SHALL show the pose name, confidence percentage, posture score (0-100), skeleton overlay, and ordered list of corrective feedback
4. IF the uploaded image does not contain a detectable person, THEN THE Dashboard SHALL display a message indicating no pose was detected

### Requirement 13: Streamlit Dashboard — Live Camera Page

**User Story:** As a yoga practitioner, I want a real-time camera view with pose detection, scoring, and feedback overlaid, so that I can practice with continuous guidance.

#### Acceptance Criteria

1. WHEN the live camera page is active, THE Dashboard SHALL capture webcam frames and run the full pipeline (classification, landmark detection, scoring, feedback) continuously
2. WHILE the camera is active, THE Dashboard SHALL display the pose name, confidence score, posture score, skeleton overlay, corrective feedback list, and voice status on the interface
3. WHEN voice feedback is enabled, THE Dashboard SHALL trigger non-blocking voice output for the top corrective instruction when feedback changes
4. IF the webcam cannot be accessed, THEN THE Dashboard SHALL display an error message and suggest using the image upload page as an alternative

### Requirement 14: Streamlit Dashboard — Model Performance Page

**User Story:** As a machine learning engineer, I want to view training metrics and evaluation results in the dashboard, so that I can assess model quality without running separate scripts.

#### Acceptance Criteria

1. WHEN the Model Performance page is loaded, THE Dashboard SHALL display overall accuracy, per-class precision, recall, and F1 score metrics
2. WHEN visualization files exist in the outputs directory, THE Dashboard SHALL display the confusion matrix plot, accuracy curve plot, and loss curve plot
3. IF metric files or visualization files are not found, THEN THE Dashboard SHALL display a message indicating the model has not been trained yet

### Requirement 15: Alternative Tabular Classification Path

**User Story:** As a developer deploying on resource-constrained hardware, I want an alternative classification method using extracted landmarks and a tabular model, so that inference can run in sub-1ms without a GPU.

#### Acceptance Criteria

1. WHEN landmarks are extracted from a frame, THE Pose_Detector SHALL produce a 132-dimensional feature vector suitable for tabular classification
2. WHEN a CSV dataset builder is invoked, THE System SHALL extract landmark features from all training images and save them with labels to a CSV file
3. WHEN the tabular model (XGBoost or Random Forest) is trained on landmark features, THE System SHALL produce a classifier capable of sub-1ms inference per sample

### Requirement 16: Performance and Latency

**User Story:** As a yoga practitioner, I want the system to run smoothly at 30 frames per second, so that feedback feels instantaneous and the video is not choppy.

#### Acceptance Criteria

1. THE System SHALL maintain full pipeline latency below 33 milliseconds per frame to sustain 30 frames per second real-time operation
2. THE Inference_Engine SHALL complete ResNet18 single-frame inference within 50 milliseconds on CPU and within 20 milliseconds on GPU
3. THE Angle_Calculator SHALL complete all 7 joint angle computations within 1 millisecond
4. THE Posture_Scorer SHALL complete score calculation within 1 millisecond
5. THE Feedback_Engine SHALL complete feedback generation within 1 millisecond

### Requirement 17: Error Handling and Graceful Degradation

**User Story:** As a user, I want the system to continue operating gracefully when components fail, so that a single error does not crash the entire application.

#### Acceptance Criteria

1. IF MediaPipe fails to detect landmarks in a frame, THEN THE System SHALL skip the scoring pipeline and display "No pose detected" without crashing
2. IF the model checkpoint file is not found, THEN THE System SHALL raise a descriptive FileNotFoundError on startup indicating the training pipeline must be run first
3. IF the webcam device cannot be opened, THEN THE System SHALL display an error in the dashboard and disable the live camera page while keeping other pages functional
4. IF one or more landmarks have visibility below 0.3, THEN THE Angle_Calculator SHALL exclude affected joints from calculations and the Posture_Scorer shall adjust weights for partial scoring
5. IF the voice engine fails to initialize, THEN THE System SHALL log a warning and continue with visual-only feedback displaying voice status as "Unavailable"
6. IF GPU memory is exhausted during inference, THEN THE System SHALL automatically fall back to CPU inference and log a performance warning

### Requirement 18: Privacy and Data Security

**User Story:** As a user, I want assurance that my webcam data is not stored or transmitted, so that my privacy is protected while using the system.

#### Acceptance Criteria

1. THE System SHALL process webcam frames in memory only and SHALL NOT save any webcam data to disk during normal operation
2. THE System SHALL validate uploaded images for format and size (maximum 10 megabytes) before processing
3. WHEN running the Streamlit dashboard, THE System SHALL maintain independent session state per browser connection
