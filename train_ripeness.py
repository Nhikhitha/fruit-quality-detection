# train_ripeness.py
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ---------- Paths ----------
BASE = os.getcwd()
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
LABELS_JSON = os.path.join(BASE, "class_labels.json")

# ---------- Parameters ----------
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20
NUM_CLASSES = 3
SEED = 123

# ---------- 1) Load datasets ----------
print("Loading train dataset from:", TRAIN_DIR)
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

print("Loading test dataset from:", TEST_DIR)
test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes (index -> name):", class_names)

# Save labels JSON (inferred order)
with open(LABELS_JSON, "w") as f:
    json.dump(class_names, f)
print("Saved labels to", LABELS_JSON)

# ---------- Cache / prefetch ----------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------- 2) Augmentation ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
], name="data_augmentation")

# ---------- 3) Build model ----------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- 4) Optional: class weights ----------
all_labels = []
for _, batch_labels in image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH,
    image_size=IMG_SIZE,
    shuffle=False
):
    all_labels.extend(np.argmax(batch_labels.numpy(), axis=1))

all_labels = np.array(all_labels)
class_weight_vals = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=all_labels,
)
class_weight = {i: w for i, w in enumerate(class_weight_vals)}
print("class_weight:", class_weight)

# ---------- 5) Callbacks ----------
checkpoint_path = "ripeness_checkpoint.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=False
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
    ),
]

# ---------- 6) Train ----------
history = model.fit(
    train_ds,
    validation_data=test_ds,  # for a real setup, split val separately
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,  # remove this line if you don’t want weights
)

# ---------- 7) Save model ----------
model.save("ripeness_cnn.keras", include_optimizer=False)
model.save("ripeness_cnn.h5", include_optimizer=False)
print("Saved models: ripeness_cnn.keras and ripeness_cnn.h5")

# ---------- 8) Evaluate on test set ----------
y_true = []
y_pred = []

for batch_imgs, batch_labels in test_ds:
    preds = model.predict(batch_imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
