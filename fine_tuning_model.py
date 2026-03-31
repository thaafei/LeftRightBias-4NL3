"""
Fine-Tuning DistilBERT for YouTube Video Category Classification
COMPSCI 4NL3 - Final Project

This script fine-tunes a DistilBERT model to classify YouTube videos into
one of 12 categories based on video title and description.

Categories: Education, Lifestyle, Sports & Games, Video Games, Food & Cooking,
Shopping, Digital Media, News, Vehicle, Music, Business, Health
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "inputData")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Category label mapping
CATEGORIES = [
    "Business",
    "Digital Media",
    "Education",
    "Food & Cooking",
    "Health",
    "Lifestyle",
    "Music",
    "News",
    "Shopping",
    "Sports & Games",
    "Vehicle",
    "Video Games",
]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}
NUM_LABELS = len(CATEGORIES)


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Loading & Preprocessing
# ============================================================
def clean_text(text):
    """Clean and preprocess text input."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove emojis and special unicode characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(data_file, label_file=None):
    """Load and merge data and label files."""
    data_df = pd.read_csv(os.path.join(DATA_DIR, data_file))

    # Clean text fields
    data_df["video_title"] = data_df["video_title"].apply(clean_text)
    data_df["video_description"] = data_df["video_description"].apply(clean_text)

    # Combine title and description: "[CLS] title [SEP] description"
    data_df["text"] = data_df["video_title"] + " [SEP] " + data_df["video_description"]

    if label_file is not None:
        label_df = pd.read_csv(os.path.join(DATA_DIR, label_file))
        data_df["category"] = label_df["category"]
        # Drop rows with missing labels
        data_df = data_df.dropna(subset=["category"]).reset_index(drop=True)
        data_df["label"] = data_df["category"].map(LABEL2ID)

    return data_df


# ============================================================
# PyTorch Dataset
# ============================================================
class YouTubeDataset(Dataset):
    """Dataset for YouTube video classification."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class YouTubeTestDataset(Dataset):
    """Dataset for test data (no labels)."""

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


# ============================================================
# Training & Evaluation
# ============================================================
def train_epoch(model, data_loader, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc="  Training", leave=False, ncols=100)
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="  Evaluating", leave=False, ncols=100)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, f1, all_preds, all_labels


def predict(model, data_loader):
    """Generate predictions for test data (no labels)."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="  Predicting", leave=False, ncols=100)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


# ============================================================
# Visualization & Analysis
# ============================================================
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_title("Training and Validation Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    ax2.set_title("Training and Validation Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {save_path}")


def error_analysis(y_true, y_pred, texts, categories, save_path):
    """Perform error analysis and save misclassified examples."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                "text": texts[i][:200],  # Truncate for readability
                "true_label": categories[y_true[i]],
                "predicted_label": categories[y_pred[i]],
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(save_path, index=False)
    print(f"Error analysis saved to {save_path} ({len(errors)} misclassified samples)")

    # Per-category error rates
    print("\n--- Per-Category Error Rate ---")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(categories))))
    for i, cat in enumerate(categories):
        total = cm[i].sum()
        if total > 0:
            error_rate = 1.0 - cm[i][i] / total
            print(f"  {cat:20s}: {error_rate:.2%} error rate ({total} samples)")

    return error_df


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LEN}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)

    # --- Load Data ---
    print("\n[1/6] Loading data...")
    train_df = load_data("training_data.csv", "training_label.csv")
    val_df = load_data("validation_data.csv", "validation_label.csv")
    test_df = load_data("testing_data.csv")

    print(f"  Training samples:   {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples:       {len(test_df)}")
    print(f"\n  Training label distribution:")
    for cat in CATEGORIES:
        count = (train_df["category"] == cat).sum()
        print(f"    {cat:20s}: {count}")

    # --- Tokenizer ---
    print(f"\n[2/6] Loading tokenizer: {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # --- Create Datasets & DataLoaders ---
    print("[3/6] Creating datasets...")
    train_dataset = YouTubeDataset(
        train_df["text"].values,
        train_df["label"].values,
        tokenizer,
        MAX_LEN,
    )
    val_dataset = YouTubeDataset(
        val_df["text"].values,
        val_df["label"].values,
        tokenizer,
        MAX_LEN,
    )
    test_dataset = YouTubeTestDataset(
        test_df["text"].values,
        tokenizer,
        MAX_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    print("[4/6] Loading pretrained model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Training Loop ---
    print(f"\n[5/6] Training for {EPOCHS} epochs...")
    print("-" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"    -> New best model saved (F1: {val_f1:.4f})")

    print("-" * 60)
    print(f"Best Validation F1: {best_val_f1:.4f}")

    # --- Load Best Model & Final Evaluation ---
    print("\n[6/6] Final evaluation with best model...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), weights_only=True))

    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Weighted F1: {val_f1:.4f}")
    print(f"\nClassification Report:")
    all_label_ids = list(range(NUM_LABELS))
    report = classification_report(
        val_labels, val_preds, labels=all_label_ids, target_names=CATEGORIES, digits=4, zero_division=0
    )
    print(report)

    # Save classification report
    report_dict = classification_report(
        val_labels, val_preds, labels=all_label_ids, target_names=CATEGORIES, output_dict=True, zero_division=0
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    # --- Confusion Matrix ---
    plot_confusion_matrix(
        val_labels,
        val_preds,
        CATEGORIES,
        "Validation Set Confusion Matrix (Fine-Tuned DistilBERT)",
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )

    # --- Training History Plot ---
    plot_training_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    # --- Error Analysis ---
    print("\n--- Error Analysis ---")
    error_df = error_analysis(
        val_labels,
        val_preds,
        val_df["text"].values,
        CATEGORIES,
        os.path.join(OUTPUT_DIR, "error_analysis.csv"),
    )

    # --- Test Set Predictions ---
    print("\nGenerating test set predictions...")
    test_preds = predict(model, test_loader)
    test_pred_labels = [ID2LABEL[p] for p in test_preds]

    test_output = pd.DataFrame({
        "video_title": test_df["video_title"],
        "predicted_category": test_pred_labels,
    })
    test_output.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"Test predictions saved to {os.path.join(OUTPUT_DIR, 'test_predictions.csv')}")

    # --- Save Training History ---
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:            {MODEL_NAME} (fine-tuned)")
    print(f"  Best Val F1:      {best_val_f1:.4f}")
    print(f"  Best Val Acc:     {val_acc:.4f}")
    print(f"  Training Samples: {len(train_df)}")
    print(f"  Val Samples:      {len(val_df)}")
    print(f"  Test Samples:     {len(test_df)}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
