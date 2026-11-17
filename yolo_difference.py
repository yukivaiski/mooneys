from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO
import pandas as pd
import seaborn as sns

original_images = Path("./original")
mooney_images = Path("./mooney")

original_model = YOLO("./best_og_yolo.pt")
mooney_model = YOLO("./best_mooney_yolo.pt")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
original_paths = sorted([
    p for p in original_images.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
])
mooney_paths = sorted([
    p for p in mooney_images.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
])

assert len(original_paths) == 100, "Expected 100 original images"
assert len(mooney_paths) == 100, "Expected 100 Mooney images"


def get_true_label(image_path):
    name = image_path.stem
    name = name.removesuffix("_mooney")
    return name.split("_")[0]


original_labels = [get_true_label(p) for p in original_paths]
mooney_labels = [get_true_label(p) for p in mooney_paths]


def evaluate_model(model, image_paths, true_labels, experiment_name):
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=224,
        batch=16,
        verbose=False
    )

    predicted_labels = [
        model.names[int(result.probs.top1)]
        for result in results
    ]

    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"\n{experiment_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return predicted_labels, accuracy


# 1. Original model → Original images
pred_oo, acc_oo = evaluate_model(
    original_model,
    original_paths,
    original_labels,
    "Original model on Original images"
)

# 2. Original model → Mooney images
pred_om, acc_om = evaluate_model(
    original_model,
    mooney_paths,
    mooney_labels,
    "Original model on Mooney images"
)

# 3. Mooney model → Original images
pred_mo, acc_mo = evaluate_model(
    mooney_model,
    original_paths,
    original_labels,
    "Mooney model on Original images"
)

# 4. Mooney model → Mooney images
pred_mm, acc_mm = evaluate_model(
    mooney_model,
    mooney_paths,
    mooney_labels,
    "Mooney model on Mooney images"
)

accuracies = [acc_oo, acc_om, acc_mo, acc_mm]

names = [
    "Original → Original",
    "Original → Mooney",
    "Mooney → Original",
    "Mooney → Mooney"
]

experiments = [
    ("Original → Original", original_labels, pred_oo),
    ("Original → Mooney", mooney_labels, pred_om),
    ("Mooney → Original", original_labels, pred_mo),
    ("Mooney → Mooney", mooney_labels, pred_mm)
]

plt.figure(figsize=(10, 6))
plt.bar(names, accuracies)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Performance Across Image Domains")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

metric_data = []

for name, true, pred in experiments:
    metric_data.extend([
        {"Experiment": name, "Metric": "Precision",
         "Score": precision_score(true, pred, average="macro", zero_division=0)},
        {"Experiment": name, "Metric": "Recall",
         "Score": recall_score(true, pred, average="macro", zero_division=0)},
        {"Experiment": name, "Metric": "F1-score",
         "Score": f1_score(true, pred, average="macro", zero_division=0)}
    ])

metrics_df = pd.DataFrame(metric_data)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=metrics_df,
    x="Experiment",
    y="Score",
    hue="Metric"
)

plt.ylim(0, 1)
plt.title("Precision, Recall and F1-score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("classification_metrics.png", dpi=300)
plt.show()

results_data = []
for name, true, pred in experiments:
    for t, p in zip(true, pred):
        results_data.append({
            "Experiment": name,
            "True breed": t,
            "Predicted breed": p,
            "Correct": t == p
        })
df_results = pd.DataFrame(results_data)

prediction_counts = (
    df_results.groupby(["Experiment", "Predicted breed"])
    .size()
    .reset_index(name="Count")
)
plt.figure(figsize=(16, 8))
sns.barplot(
    data=prediction_counts,
    x="Predicted breed",
    y="Count",
    hue="Experiment"
)
plt.title("Distribution of Predicted Breeds")
plt.ylabel("Number of predictions")
plt.xlabel("Predicted breed")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("prediction_distribution.png", dpi=300)
plt.show()