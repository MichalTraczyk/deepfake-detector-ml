from keras.src.applications.xception import Xception
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


image_size = (256, 256)
batch_size = 32

data_dir = "../../data_split/"
def add_fft_to_dataset(rgb_dataset):
    def map_fn(image, label):
        # image: shape (256, 256, 3), dtype float32 [0, 255]
        rgb = tf.cast(image, tf.float32)
        gray = tf.image.rgb_to_grayscale(rgb)

        # FFT
        fft = tf.signal.fft2d(tf.cast(gray[..., 0], tf.complex64))
        fft_shifted = tf.signal.fftshift(fft)
        magnitude = tf.math.log(tf.abs(fft_shifted) + 1e-8)
        magnitude = tf.expand_dims(magnitude, axis=-1)  # shape (256, 256, 1)
        magnitude = tf.image.per_image_standardization(magnitude)

        inputs = {
            "rgb_input": rgb,
            "fft_input": magnitude
        }
        return inputs, label

    return rgb_dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

def build_model():
    # FFT input
    input_fft = layers.Input(shape=(256, 256, 1), name="fft_input")
    x_fft = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_fft)
    x_fft = layers.MaxPooling2D((2, 2))(x_fft)
    x_fft = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x_fft)
    x_fft = layers.GlobalAveragePooling2D()(x_fft)

    # RGB input
    input_rgb = layers.Input(shape=(256, 256, 3), name="rgb_input")
    x_rgb = layers.Rescaling(1. / 255)(input_rgb)

    # Xception feature extractor
    base_model = Xception(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
    base_model.trainable = False
    x_rgb = base_model(x_rgb, training=False)
    x_rgb = layers.GlobalAveragePooling2D()(x_rgb)

    # Combine both
    combined = layers.Concatenate()([x_rgb, x_fft])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    outputs = layers.Dense(1, activation='sigmoid')(combined)

    # Create final model
    model = models.Model(inputs={"rgb_input": input_rgb, "fft_input": input_fft}, outputs=outputs)
    return model


test_ds = image_dataset_from_directory(
    data_dir + "test",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size
)
test_ds = add_fft_to_dataset(test_ds)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
model.load_weights("saved/full_model.keras")

# Evaluate using built-in metrics
results = model.evaluate(test_ds)
print(f"\nBuilt-in Metrics:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Predict and calculate advanced metrics
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = np.concatenate([y for _, y in test_ds], axis=0).astype(int).flatten()

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))


# Ensure y_pred_probs is 1D
y_scores = y_pred_probs.flatten()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='green', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.tight_layout()
plt.show()