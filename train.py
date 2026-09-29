"""
train.py
Step 2 of the project. Run it with:  python train.py

What it does:
  Stage 1  Keeps MobileNetV2 (already trained on ImageNet) frozen and trains only a new
           final layer that recognises your 4 road damage classes.
  Stage 2  "Fine-tunes": unfreezes the last few MobileNetV2 layers and trains them very
           gently, so the model adapts its features to road textures.
  Finally  Tests the best model on images it has never seen and saves charts to results/.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's startup noise

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import config

AUTOTUNE = tf.data.AUTOTUNE
preprocess_input = keras.applications.mobilenet_v2.preprocess_input  # scales pixels to [-1, 1]


# ---------------------------------------------------------------- data
def load_split(split_name, shuffle):
    return keras.utils.image_dataset_from_directory(
        config.PROCESSED_DIR / split_name,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=config.SEED,
    )


def prepare(dataset, cache=True):
    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if cache:  # keep in memory for speed (not for training data, so it reshuffles every epoch)
        dataset = dataset.cache()
    return dataset.prefetch(AUTOTUNE)


# ---------------------------------------------------------------- model
def build_model(num_classes, weights="imagenet"):
    """MobileNetV2 body + a small new 'head' for our classes.
    Built as one flat model so Grad-CAM can later reach the 'out_relu' layer."""
    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    base = keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights=weights)
    base.trainable = False  # Stage 1: freeze everything ImageNet taught it

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)  # randomly drops features while training, reduces overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="road_damage_mobilenetv2"), base


def unfreeze_top(base):
    base.trainable = True
    for layer in base.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    # BatchNorm layers stay frozen; retraining them on a small dataset hurts accuracy
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def callbacks(best_so_far=None):
    return [
        # Saves the model only when validation loss improves
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True,
            initial_value_threshold=best_so_far, verbose=1,
        ),
        # Stops early if validation loss hasn't improved for 5 epochs
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        # Lowers the learning rate when progress stalls
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ]


# ---------------------------------------------------------------- charts and evaluation
def plot_history(histories, fine_tune_start):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, train_vals, val_vals, title in [
        (axes[0], acc, val_acc, "Accuracy"),
        (axes[1], loss, val_loss, "Loss"),
    ]:
        ax.plot(train_vals, label="Training")
        ax.plot(val_vals, label="Validation")
        ax.axvline(fine_tune_start - 0.5, color="grey", linestyle="--", label="Fine-tuning starts")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def evaluate(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3,
                                   zero_division=0)
    (config.RESULTS_DIR / "test_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
    )
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return report, float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------- main
def main(weights="imagenet"):
    if not (config.PROCESSED_DIR / "train").exists():
        raise SystemExit("No processed data found. Run  python prepare_data.py  first.")
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    keras.utils.set_random_seed(config.SEED)

    train_raw = load_split("train", shuffle=True)
    class_names = train_raw.class_names
    train_ds = prepare(train_raw, cache=False)
    val_ds = prepare(load_split("val", shuffle=False))
    test_ds = prepare(load_split("test", shuffle=False))

    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"\nClasses: {class_names}")

    model, base = build_model(len(class_names), weights=weights)

    print("\n=== Stage 1: training the new classifier layer ===")
    compile_model(model, config.LR_HEAD)
    history_1 = model.fit(train_ds, validation_data=val_ds,
                          epochs=config.EPOCHS_HEAD, callbacks=callbacks())
    best_val_loss = min(history_1.history["val_loss"])

    print(f"\n=== Stage 2: fine-tuning the last {config.FINE_TUNE_LAYERS} MobileNetV2 layers ===")
    unfreeze_top(base)
    compile_model(model, config.LR_FINE_TUNE)  # must re-compile after changing what's trainable
    history_2 = model.fit(train_ds, validation_data=val_ds,
                          epochs=config.EPOCHS_FINE_TUNE, callbacks=callbacks(best_val_loss))

    plot_history([history_1, history_2], fine_tune_start=len(history_1.history["loss"]))

    print("\n=== Testing the best saved model on unseen images ===")
    best_model = keras.models.load_model(config.MODEL_PATH)
    report, accuracy = evaluate(best_model, test_ds, class_names)
    print(report)
    print(f"Test accuracy: {accuracy:.1%}")
    print(f"\nModel saved to   {config.MODEL_PATH}")
    print(f"Charts saved to  {config.RESULTS_DIR}")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
