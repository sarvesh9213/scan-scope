"""
Training script for the Chest X-Ray Pneumonia classifier.

Dataset: Kermany et al. "Chest X-Ray Images (Pneumonia)"
  - Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
  - Also on Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - License: CC BY 4.0

Expected directory structure (Kaggle format):
    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Usage:
    python training/train_chest_xray.py --data_dir /path/to/chest_xray

This script:
  1. Loads training/validation/test splits as provided by the dataset.
  2. Fine-tunes a ResNet-18 (pretrained on ImageNet) for binary classification.
  3. Evaluates on the held-out test set with multiple metrics.
  4. Saves model weights and evaluation results.
  5. Records training configuration, random seed, and class distribution.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(augment=False):
    """Get preprocessing transforms."""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(num_classes=2, pretrained=True):
    """Build a ResNet-18 model for binary classification."""
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)} — "
                  f"Loss: {loss.item():.4f} — "
                  f"Acc: {100. * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_test_set(model, loader, device, class_names):
    """
    Full evaluation on the test set.

    Returns a dict with accuracy, precision, recall, specificity,
    F1, ROC-AUC, confusion matrix, and class distribution.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    model.eval()
    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # Score for the positive class (PNEUMONIA = class 1)
            all_scores.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_scores = np.array(all_scores)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Specificity (TNR)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = None

    # Class distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = {}
    for idx, count in zip(unique, counts):
        class_dist[class_names[idx]] = int(count)

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(specificity),
        'f1': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': class_names,
        'class_distribution': class_dist,
        'test_set_size': len(y_true),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Chest X-Ray Pneumonia Classifier"
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to chest_xray dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Paths
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            sys.exit(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets
    # The ImageFolder class assigns class indices alphabetically:
    # NORMAL=0, PNEUMONIA=1
    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=get_transforms(augment=True)
    )
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=get_transforms(augment=False)
    )
    test_dataset = datasets.ImageFolder(
        str(test_dir),
        transform=get_transforms(augment=False)
    )

    class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']
    print(f"Classes: {class_names}")
    print(f"Class-to-index mapping: {train_dataset.class_to_idx}")

    # Log class distribution
    train_dist = {}
    for _, label in train_dataset.samples:
        name = class_names[label]
        train_dist[name] = train_dist.get(name, 0) + 1
    print(f"Training set distribution: {train_dist}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Model
    model = build_model(num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training config (saved for reproducibility)
    training_config = {
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'architecture': 'ResNet-18',
        'pretrained': 'ImageNet (torchvision)',
        'augmentation': [
            'RandomCrop(224 from 256)',
            'RandomHorizontalFlip',
            'RandomRotation(10)',
            'RandomAffine(translate=0.05)'
        ],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'train_class_distribution': train_dist,
        'patience': args.patience,
        'device': str(device),
    }

    # Output directory
    weights_dir = os.path.join(PROJECT_ROOT, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  LR: {args.lr}, Patience: {args.patience}")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s\n")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(weights_dir, 'chest_xray_resnet18.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement "
                      f"for {args.patience} epochs)")
                break

    # Reload best model for evaluation
    best_path = os.path.join(weights_dir, 'chest_xray_resnet18.pth')
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print(f"\nLoaded best model from {best_path}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("  Evaluating on held-out test set...")
    print(f"{'='*60}\n")

    metrics = evaluate_test_set(model, test_loader, device, class_names)

    # Print metrics
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    {'':>15} Pred:{class_names[0]:>10} Pred:{class_names[1]:>10}")
    for i, row in enumerate(cm):
        print(f"    True:{class_names[i]:>10} {row[0]:>10} {row[1]:>10}")
    print(f"\n  Test set: {metrics['class_distribution']}")

    # Save metrics
    metrics_path = os.path.join(weights_dir, 'chest_xray_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Saved evaluation metrics to {metrics_path}")

    # Save training config
    training_config['best_val_acc'] = best_val_acc
    training_config['history'] = history
    config_path = os.path.join(weights_dir, 'chest_xray_training_config.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    print(f"  ✓ Saved training config to {config_path}")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
