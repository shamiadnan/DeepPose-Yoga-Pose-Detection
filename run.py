"""
DeepPose 2.0 — Main Entry Point
=================================
Quick launcher for common operations.
Usage:
    python run.py train      # Train the model
    python run.py evaluate   # Evaluate trained model
    python run.py app        # Launch Streamlit dashboard
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("DeepPose 2.0 — AI-Powered Yoga Pose Recognition")
        print("=" * 50)
        print("\nUsage:")
        print("  python run.py train      Train the ResNet18 model")
        print("  python run.py evaluate   Evaluate the trained model")
        print("  python run.py app        Launch Streamlit dashboard")
        print("\nFor more info, see README.md")
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        from models.train import DatasetManager, YogaTrainer
        import config
        
        print("🧘 DeepPose 2.0 — Training Pipeline")
        print("=" * 50)
        
        manager = DatasetManager(config.DATASET_DIR, config.IMG_SIZE, config.BATCH_SIZE)
        train_loader, val_loader, test_loader = manager.load_and_split(
            test_size=config.TEST_SIZE, val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE
        )
        class_weights = manager.get_class_weights()
        
        trainer = YogaTrainer(
            num_classes=config.NUM_CLASSES,
            learning_rate=config.LEARNING_RATE,
            patience=config.PATIENCE
        )
        history = trainer.train(
            train_loader, val_loader, 
            epochs=config.EPOCHS, class_weights=class_weights
        )
        trainer.save_checkpoint()
        
        # Auto-evaluate after training
        from models.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves
        
        metrics = evaluate_model(trainer.model, test_loader, manager.get_class_names())
        plot_confusion_matrix(metrics['confusion_matrix'], manager.get_class_names())
        plot_training_curves(history)
        
        print("\n✅ Training complete! Run 'python run.py app' to launch dashboard.")
    
    elif command == "evaluate":
        from models.train import DatasetManager, YogaTrainer
        from models.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves
        import config
        import torch
        from torchvision import models
        import torch.nn as nn
        
        print("🧘 DeepPose 2.0 — Model Evaluation")
        print("=" * 50)
        
        if not os.path.exists(config.MODEL_PATH):
            print(f"❌ Model not found at {config.MODEL_PATH}")
            print("Run 'python run.py train' first.")
            return
        
        # Load model
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model = model.to(config.DEVICE)
        model.eval()
        
        # Load test data
        manager = DatasetManager(config.DATASET_DIR, config.IMG_SIZE, config.BATCH_SIZE)
        _, _, test_loader = manager.load_and_split()
        
        metrics = evaluate_model(model, test_loader, manager.get_class_names())
        plot_confusion_matrix(metrics['confusion_matrix'], manager.get_class_names())
        
        print("\n✅ Evaluation complete! Check outputs/ directory for plots.")
    
    elif command == "app":
        print("🧘 DeepPose 2.0 — Launching Dashboard...")
        print("=" * 50)
        os.system("streamlit run streamlit_app/app.py")
    
    else:
        print(f"❌ Unknown command: '{command}'")
        print("Available: train, evaluate, app")


if __name__ == "__main__":
    main()
