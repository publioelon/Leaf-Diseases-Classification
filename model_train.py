import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_classes_and_counts(directory):
    class_names = sorted(os.listdir(directory))
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}
    class_counts = {class_name: len(glob(os.path.join(directory, class_name, '*.png'))) for class_name in class_names}
    return class_names, class_indices, class_counts

def load_and_process_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (target_size[0], target_size[1]))
    return image.astype('float32') / 255.0 if image is not None else None

def create_generator(directory, class_indices, batch_size, datagen, target_size):
    image_paths = glob(os.path.join(directory, '*/*.png'))
    np.random.shuffle(image_paths)
    while True:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [load_and_process_image(x, target_size) for x in batch_paths if load_and_process_image(x, target_size) is not None]
            labels = [class_indices[os.path.basename(os.path.dirname(x))] for x in batch_paths if load_and_process_image(x, target_size) is not None]
            if images:
                yield np.array(images), tf.keras.utils.to_categorical(labels, num_classes=len(class_indices))

def compute_class_weights(labels):
    classes = np.unique(labels)
    class_weights = {}
    for cls in classes:
        class_weights[cls] = len(labels) / (len(classes) * np.sum(labels == cls))
    return class_weights

def plot_learning_curves(history, model_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(f'{model_name}_learning_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def setup_model(model_name, input_shape):
    batch_size, learning_rate, epochs, datagen_params, dense_units, dropout_rate, reg = None, None, None, None, None, None, None

    if model_name == 'MobileNetV1':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        batch_size, learning_rate, epochs = 32, 1e-4, 5
        dense_units, dropout_rate, reg = 128, 0.5, l2(0.01)
        datagen_params = {'rotation_range': 90}

    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        batch_size, learning_rate, epochs = 128, 5e-6, 120
        dense_units, dropout_rate = [256, 128], 0.5
        datagen_params = {'rotation_range': 90, 'shear_range': 0.2, 'zoom_range': 0.2, 'horizontal_flip': True}

    elif model_name == 'VGG-16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        batch_size, learning_rate, epochs = 32, 5e-5, 25
        dropout_rate = 0.4
        datagen_params = {'rotation_range': 20, 'width_shift_range': 0.1, 'height_shift_range': 0.1, 'zoom_range': 0.1, 'horizontal_flip': True}

    elif model_name == 'InceptionV3':
        input_shape = (229, 229, 3)  # Adjust input shape for InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        batch_size, learning_rate, epochs = 16, 5e-5, 10
        dense_units, dropout_rate = 16, 0.35
        datagen_params = {'rotation_range': 90, 'width_shift_range': 0.25, 'height_shift_range': 0.25, 'shear_range': 0.25, 'zoom_range': 0.25, 'brightness_range': [0.5, 1.5], 'horizontal_flip': True}

    datagen = ImageDataGenerator(**datagen_params)
    x = GlobalAveragePooling2D()(base_model.output)

    if isinstance(dense_units, list):  # For models with multiple dense layers
        for units in dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
    else:
        x = Dense(dense_units, activation='relu', kernel_regularizer=reg if model_name == 'MobileNetV1' else None)(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(7, activation='softmax')(x)  # Using 7 for the number of classes
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, datagen, batch_size, epochs, input_shape

def main(model_name):
    data_dir = '/mnt/c/Users/Publi/Downloads/archive/split_data'
    train_dir, val_dir, test_dir = [os.path.join(data_dir, x) for x in ['train', 'validation', 'test']]
    input_shape = (224, 224, 3)  # Default input shape for most models

    model, datagen, batch_size, epochs, input_shape = setup_model(model_name, input_shape)

    train_classes, train_indices, train_counts = get_classes_and_counts(train_dir)
    all_labels = []
    for class_name, idx in train_indices.items():
        count = train_counts[class_name]
        all_labels += [idx] * count
    class_weights = compute_class_weights(np.array(all_labels))
    class_weight_dict = dict(zip(np.unique(all_labels), class_weights))

    train_generator = create_generator(train_dir, train_indices, batch_size, datagen, input_shape)
    val_generator = create_generator(val_dir, train_indices, batch_size, datagen, input_shape)
    test_generator = create_generator(test_dir, train_indices, batch_size, datagen, input_shape)

    print(f"Starting training with {model_name}")
    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=len(glob(os.path.join(train_dir, '*/*.png'))) // batch_size, validation_data=val_generator, validation_steps=len(glob(os.path.join(val_dir, '*/*.png'))) // batch_size, class_weight=class_weight_dict)
    plot_learning_curves(history, model_name)

    # Saving the model
    model_save_path = os.path.join('saved_models', model_name)
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    test_images, test_labels = next(test_generator)
    test_predictions = model.predict(test_images)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = np.argmax(test_labels, axis=1)
    plot_confusion_matrix(test_true_labels, test_predicted_labels, train_classes, model_name)

    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(glob(os.path.join(test_dir, '*/*.png'))) // batch_size)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python this_script.py <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    main(model_name)
