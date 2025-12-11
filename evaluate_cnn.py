import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# -----------------------------
# Load model + labels
# -----------------------------
model = tf.keras.models.load_model("ripeness_cnn.keras", compile=False)

labels = ["overripe", "ripe", "unripe"]

# -----------------------------
# Load TEST Dataset
# -----------------------------
test_dir = "test"   # folder with 3 subfolders: overripe, ripe, unripe

img_size = (224, 224)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
    batch_size=32,
    image_size=img_size
)

y_true = np.argmax(np.vstack([y for x, y in test_ds]), axis=1)

# -----------------------------
# Predict
# -----------------------------
y_pred_logits = model.predict(test_ds)
y_pred = np.argmax(y_pred_logits, axis=1)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_pred == y_true)
print("\nTest Accuracy:", accuracy)

# -----------------------------
# Classification Report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=labels)
print("\nClassification Report:\n", report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ripeness CNN")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")
plt.show()

print("\nSaved: cnn_confusion_matrix.png")
