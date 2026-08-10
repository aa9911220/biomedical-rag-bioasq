import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


PRED_PATH = "results/dense_top5_predictions.json"


# Load predictions
with open(PRED_PATH, "r") as f:
    data = json.load(f)


# Extract labels
predictions = [
    item["prediction"].lower().strip()
    for item in data
]

labels = [
    item["label"].lower().strip()
    for item in data
]


# Only keep binary labels
valid_labels = {"yes", "no"}

filtered_predictions = []
filtered_labels = []

for pred, label in zip(predictions, labels):
    if label in valid_labels:
        filtered_labels.append(label)
        filtered_predictions.append(pred)


# Metrics
accuracy = accuracy_score(
    filtered_labels,
    filtered_predictions
)

macro_f1 = f1_score(
    filtered_labels,
    filtered_predictions,
    labels=["yes", "no"],
    average="macro"
)


print("====================")
print(f"Samples: {len(filtered_labels)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("====================")


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        filtered_labels,
        filtered_predictions,
        labels=["yes", "no"],
        zero_division=0
    )
)


# Confusion matrix
print("\nConfusion Matrix:")
print(
    "        Pred yes  Pred no"
)

cm = confusion_matrix(
    filtered_labels,
    filtered_predictions,
    labels=["yes", "no"]
)

print(
    f"True yes   {cm[0][0]:5d}     {cm[0][1]:5d}"
)

print(
    f"True no    {cm[1][0]:5d}     {cm[1][1]:5d}"
)
