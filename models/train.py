"""
DeepPose 2.0 — Training Pipeline
=================================
Contains DatasetManager for data loading/splitting and YogaTrainer for 
ResNet18 transfer learning with early stopping.
"""

import os
import sys
from typing import Tuple, List, Dict
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class YogaDataset(Dataset):
    """Custom PyTorch Dataset for yoga pose images.
    
    Args:
        image_paths: List of absolute paths to image files
        labels: List of integer labels corresponding to each image
        transform: Optional torchvision transforms to apply
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DatasetManager:
    """Manages dataset loading, splitting, augmentation, and DataLoader creation.
    
    Handles the complete data pipeline from raw images to training-ready DataLoaders
    with stratified splitting and class imbalance handling.
    
    Args:
        dataset_path: Path to root dataset directory containing class subdirectories
        img_size: Target image size for resizing (default: 224 for ResNet18)
        batch_size: Batch size for DataLoaders (default: 32)
    
    Example:
        >>> manager = DatasetManager("dataset/", img_size=224, batch_size=32)
        >>> train_loader, val_loader, test_loader = manager.load_and_split()
        >>> class_weights = manager.get_class_weights()
    """
    
    def __init__(self, dataset_path: str, img_size: int = 224, 
                 batch_size: int = 32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Will be populated after load_and_split()
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        self.train_labels: List[int] = []
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
        ])
    
    def _discover_dataset(self) -> None:
        """Walk dataset directory to discover all images and their labels.
        
        Expects directory structure:
            dataset_path/
                ClassName1/
                    image1.jpg
                    image2.png
                ClassName2/
                    ...
        
        Raises:
            ValueError: If dataset path doesn't exist or classes are missing
        """
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Get sorted class directories
        class_dirs = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])
        
        if len(class_dirs) == 0:
            raise ValueError(
                f"No class subdirectories found in: {self.dataset_path}"
            )
        
        self.class_names = class_dirs
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.dataset_path, class_name)
            class_images = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            if len(class_images) == 0:
                raise ValueError(
                    f"No valid images found for class '{class_name}' in: {class_dir}"
                )
            
            self.image_paths.extend(class_images)
            self.labels.extend([class_idx] * len(class_images))
        
        print(f"[DatasetManager] Discovered {len(self.image_paths)} images "
              f"across {len(self.class_names)} classes")
        for i, name in enumerate(self.class_names):
            count = self.labels.count(i)
            print(f"  {name}: {count} images")
    
    def load_and_split(self, test_size: float = 0.2, val_size: float = 0.1,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load dataset and perform stratified train/val/test split.
        
        Uses sklearn's train_test_split with stratify parameter to ensure
        each split preserves the original class distribution.
        
        Args:
            test_size: Proportion of data for test set (default: 0.2)
            val_size: Proportion of data for validation set (default: 0.1)
            random_state: Random seed for reproducibility (default: 42)
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader) DataLoaders
        
        Raises:
            ValueError: If dataset is empty or improperly structured
        """
        # Discover dataset
        self._discover_dataset()
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = \
            train_test_split(
                self.image_paths, self.labels,
                test_size=test_size,
                random_state=random_state,
                stratify=self.labels
            )
        
        # Second split: separate validation from training
        # Adjust val_size relative to train_val set
        relative_val_size = val_size / (1.0 - test_size)
        train_paths, val_paths, train_labels, val_labels = \
            train_test_split(
                train_val_paths, train_val_labels,
                test_size=relative_val_size,
                random_state=random_state,
                stratify=train_val_labels
            )
        
        # Store train labels for class weight computation
        self.train_labels = train_labels
        
        # Create datasets with appropriate transforms
        train_dataset = YogaDataset(train_paths, train_labels, self.train_transform)
        val_dataset = YogaDataset(val_paths, val_labels, self.eval_transform)
        test_dataset = YogaDataset(test_paths, test_labels, self.eval_transform)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        print(f"\n[DatasetManager] Split complete:")
        print(f"  Train: {len(train_paths)} images")
        print(f"  Validation: {len(val_paths)} images")
        print(f"  Test: {len(test_paths)} images")
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self) -> List[str]:
        """Returns sorted list of class names discovered from dataset.
        
        Returns:
            List of class name strings
        """
        return self.class_names
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for handling class imbalance.
        
        Weights are calculated as: total_samples / (num_classes * class_count)
        This gives higher weight to under-represented classes.
        
        Returns:
            torch.Tensor of shape (num_classes,) with class weights
        
        Raises:
            ValueError: If load_and_split() hasn't been called yet
        """
        if not self.train_labels:
            raise ValueError(
                "Must call load_and_split() before get_class_weights()"
            )
        
        counter = Counter(self.train_labels)
        total = len(self.train_labels)
        num_classes = len(self.class_names)
        
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 1)  # Avoid division by zero
            weight = total / (num_classes * count)
            weights.append(weight)
        
        weights_tensor = torch.FloatTensor(weights)
        print(f"\n[DatasetManager] Class weights computed:")
        for i, name in enumerate(self.class_names):
            print(f"  {name}: {weights[i]:.3f}")
        
        return weights_tensor


class YogaTrainer:
    """Manages ResNet18 transfer learning with early stopping.
    
    Loads a pretrained ResNet18, freezes the convolutional backbone,
    replaces the final FC layer for 11-class classification, and trains
    with weighted cross-entropy loss and early stopping.
    
    Args:
        num_classes: Number of output classes (default: 11)
        learning_rate: Adam optimizer learning rate (default: 0.001)
        patience: Early stopping patience in epochs (default: 5)
        device: Training device 'cuda' or 'cpu' (default: from config)
    
    Example:
        >>> trainer = YogaTrainer(num_classes=11, patience=5)
        >>> history = trainer.train(train_loader, val_loader, epochs=50)
        >>> trainer.save_checkpoint("checkpoints/best_model.pth")
    """
    
    def __init__(self, num_classes: int = 11, learning_rate: float = 0.001,
                 patience: int = 5, device: str = None):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device or config.DEVICE
        
        # Build model
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        
        # Best model state for checkpoint
        self.best_model_state = None
    
    def build_model(self) -> nn.Module:
        """Create ResNet18 with frozen backbone and new classification head.
        
        - Loads ResNet18 pretrained on ImageNet (IMAGENET1K_V1 weights)
        - Freezes all convolutional layers (no gradient updates)
        - Replaces final fc layer: Linear(512 → num_classes)
        
        Returns:
            nn.Module: Modified ResNet18 model
        """
        # Load pretrained ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final FC layer (unfrozen by default)
        num_features = model.fc.in_features  # 512
        model.fc = nn.Linear(num_features, self.num_classes)
        
        print(f"[YogaTrainer] Model built:")
        print(f"  Backbone: ResNet18 (frozen)")
        print(f"  Classifier: Linear({num_features} → {self.num_classes})")
        print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, class_weights: torch.Tensor = None) -> Dict[str, List[float]]:
        """Full training loop with early stopping.
        
        Trains the model using CrossEntropyLoss (weighted) and Adam optimizer.
        Implements early stopping: if validation loss doesn't improve for
        `patience` consecutive epochs, training stops and best weights are retained.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum number of training epochs (default: 50)
            class_weights: Optional class weights tensor for loss function
        
        Returns:
            Dictionary with keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            Each value is a list of per-epoch metrics.
        """
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer: only train the FC layer parameters
        optimizer = torch.optim.Adam(
            self.model.fc.parameters(), lr=self.learning_rate
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n[YogaTrainer] Starting training:")
        print(f"  Epochs: {epochs}, Patience: {self.patience}")
        print(f"  LR: {self.learning_rate}, Device: {self.device}")
        print(f"  {'='*50}")
        
        for epoch in range(epochs):
            # ---- Training Phase ----
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # ---- Validation Phase ----
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # ---- Early Stopping Check ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
                print(f"         ✓ Best model saved (val_loss improved to {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"         ✗ No improvement ({patience_counter}/{self.patience})")
                
                if patience_counter >= self.patience:
                    print(f"\n  [Early Stopping] Training stopped at epoch {epoch+1}")
                    break
        
        # Load best model weights
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n[YogaTrainer] Best model restored (val_loss: {best_val_loss:.4f})")
        
        return history
    
    def save_checkpoint(self, path: str = None) -> None:
        """Save model state dictionary to disk.
        
        Args:
            path: File path for checkpoint (default: config.MODEL_PATH)
        """
        if path is None:
            path = config.MODEL_PATH
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[YogaTrainer] Checkpoint saved to: {path}")
