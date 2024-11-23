import tensorflow as tf
import os
import rasterio
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

warnings.filterwarnings("ignore")

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Error setting memory growth:", e)

image_size=(128,128)

def load_tif_image(file_path):
    # Convert to string
    file_path = file_path.numpy().decode("utf-8")
    with rasterio.open(file_path) as src:
        bands = [src.read(band, out_shape=(image_size[1], image_size[0])) for band in range(1, 14)]
    hsi_image = np.stack(bands, axis=-1).astype(np.float32)  # Shape: (height, width, 13)
    return hsi_image

def load_and_augment_image(file_path,label=0):
    hsi_image = tf.py_function(func=load_tif_image, inp=[file_path], Tout=tf.float32)
    hsi_image = tf.reshape(hsi_image, [image_size[1], image_size[0], 13])
    return hsi_image, label




if __name__ == "__main__":

    # Define the path to your dataset and desired image size
    data_dir = r"C:\Users\jvkch\OneDrive\Desktop\ECEN\project\EuroSAT_MS\test"

    # Map class names to numeric labels
    class_names = sorted(os.listdir(data_dir))
    class_to_label = {name: i for i, name in enumerate(class_names)}

    # Generate file paths and labels
    file_paths = []
    labels = []
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        label = class_to_label[class_name]
        for file_name in os.listdir(class_path):
            if file_name.endswith(".tif"):
                file_paths.append(os.path.join(class_path, file_name))
                labels.append(label)

    # Create TensorFlow dataset from file paths and labels
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Load, augment, and batch dataset
    test_ds = dataset.shuffle(len(file_paths), seed=42)

    test_ds = dataset.take(len(test_ds)).map(load_and_augment_image).batch(8).prefetch(tf.data.AUTOTUNE)

    # Load the saved model
    model = tf.keras.models.load_model(r'C:\Users\jvkch\OneDrive\Desktop\ECEN\project\EuroSAT_MS\tensorflow\model\best_model.h5')

    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    true_labels = []
    predicted_labels = []

    for images, labels in test_ds:
        # Predict class probabilities
        predictions = model.predict(images)
        # Get predicted class labels
        predicted_classes = np.argmax(predictions, axis=1)
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predicted_classes.tolist())

    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Step 5: Create a confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.show()

    precision = []
    recall = []
    f1_score = []

    for i in range(len(class_names)):
        # True Positives
        tp = conf_matrix[i, i]
        # False Positives: Sum of column i excluding tp
        fp = conf_matrix[:, i].sum() - tp
        # False Negatives: Sum of row i excluding tp
        fn = conf_matrix[i, :].sum() - tp

        # Precision: tp / (tp + fp)
        if tp + fp > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(0.0)

        # Recall: tp / (tp + fn)
        if tp + fn > 0:
            recall.append(tp / (tp + fn))
        else:
            recall.append(0.0)

        # F1-Score: 2 * (precision * recall) / (precision + recall)
        if precision[-1] + recall[-1] > 0:
            f1_score.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
        else:
            f1_score.append(0.0)

    # Step 8: Overall Metrics
    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)
    overall_f1_score = np.mean(f1_score)

    # Step 7: Create a DataFrame for the metrics
    metrics_data = {
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

    # Create a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Calculate and add overall metrics
    overall_metrics = pd.DataFrame({
        "Class": ["Overall"],
        "Precision": [overall_precision],
        "Recall": [overall_recall],
        "F1-Score": [overall_f1_score]
    })

    # Append overall metrics to the table
    final_metrics_df = pd.concat([metrics_df, overall_metrics], ignore_index=True)

    print(final_metrics_df)

