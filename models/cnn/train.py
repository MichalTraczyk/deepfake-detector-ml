import cv2
import numpy as np
from keras.src.applications.xception import Xception
import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
from keras.src.callbacks import ModelCheckpoint

image_size = (256, 256)
batch_size = 32

data_dir = "../../data_split/"
# Load datasets
train_ds = image_dataset_from_directory(
    data_dir + "train",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    data_dir + "val",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    data_dir + "test",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size
)
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


train_ds = add_fft_to_dataset(train_ds)
val_ds = add_fft_to_dataset(val_ds)
test_ds = add_fft_to_dataset(test_ds)

# Improve performance with prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Checkpoint callback
checkpoint_path = "saved/checkpoint.weights.h5"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)


def compute_fft_input(image):
    image = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    return magnitude[..., np.newaxis]  # shape: (224, 224, 1)


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


model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = 10
resume = int(input("Resume? How many more epochs, 0 if fresh train: "))
if resume != 0:
    model.load_weights(checkpoint_path)
    epochs = resume

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb]
)

model.save("saved/full_model.keras")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")
