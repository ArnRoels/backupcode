import os
import shutil
import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config
from model import QualityModel

# Load the model
model = QualityModel().to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH, weights_only=True))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

# Load the test datasets
test_good_dataset = ImageDataset(Config.TEST_GOOD, transform=transform, label=1)
test_bad_dataset = ImageDataset(Config.TEST_BAD, transform=transform, label=0)

# Create data loaders
test_good_loader = DataLoader(test_good_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
test_bad_loader = DataLoader(test_bad_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Function to evaluate the model
def evaluate_model(loader, label):
    all_preds = []
    all_labels = []
    wrong_preds = []
    with torch.no_grad():
        for images, _, filenames in loader:
            for img_tensor, filename in zip(images, filenames):
                # Only evaluate images ending with _C1.tiff
                if not filename.endswith("_C1.tiff"):
                    continue

                img_tensor = img_tensor.to(Config.DEVICE)
                prediction = model(img_tensor.unsqueeze(0)).item()

                # Convert prediction to binary class
                pred_label = 1 if prediction >= 0.5 else 0
                all_preds.append(pred_label)
                all_labels.append(label)

                # Check if the prediction is wrong
                if pred_label != label:
                    wrong_preds.append(filename)
    return all_preds, all_labels, wrong_preds

# Evaluate on good and bad test sets
good_preds, good_labels, good_wrong_preds = evaluate_model(test_good_loader, 1)
bad_preds, bad_labels, bad_wrong_preds = evaluate_model(test_bad_loader, 0)

# Combine results
all_preds = good_preds + bad_preds
all_labels = good_labels + bad_labels
wrong_preds = good_wrong_preds + bad_wrong_preds

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print wrongfully predicted samples
print("Wrongfully predicted samples:")
for filename in wrong_preds:
    print(filename)