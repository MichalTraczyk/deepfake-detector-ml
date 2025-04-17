import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
from keras.src.callbacks import ModelCheckpoint
image_size = (256, 256)
batch_size = 32

train_ds = image_dataset_from_directory(
    "data_processed/Train",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    "data_processed/Validation",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    "data_processed/Test",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

checkpoint_path = "checkpoints/checkpoint.weights.h5"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)


# Build model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(256, 256, 3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
epochs = 10
resume = int(input("Resume? How many more epochs, 0 if fresh train"))
if resume != 0:
    model.load_weights("checkpoints/checkpoint.weights.h5")
    epochs = resume
# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb]
)

model.save("saved_model/full_model.keras")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")