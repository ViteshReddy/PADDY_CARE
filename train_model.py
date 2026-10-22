
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_images"
MODEL_DIR = BASE_DIR / "models"

# Training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training images...")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    print("\nDetected classes:")
    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    # Save class names in the same order used by the dataset
    with open(MODEL_DIR / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=4)

    # Improve input-pipeline performance
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    print("\nLoading MobileNetV2 pretrained image model...")

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(
        len(class_names),
        activation="softmax",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model_path = MODEL_DIR / "paddy_disease_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the final model weights as well, using the same file path.
    # The checkpoint above already saves the best validation model.
    print("\nTraining finished.")
    print(f"Model saved to: {model_path}")
    print(f"Class names saved to: {MODEL_DIR / 'class_names.json'}")

    if history.history.get("val_accuracy"):
        best_accuracy = max(history.history["val_accuracy"])
        print(f"Best validation accuracy: {best_accuracy:.2%}")

if __name__ == "__main__":
    main()
