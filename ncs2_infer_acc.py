import openvino.runtime as ov
import os
import cv2
import numpy as np
import glob
import time
from sklearn.metrics import accuracy_score
import statistics

def normalize(image):
    return image / 255.0

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Image not found at path: {image_path}")
    image = cv2.resize(image, (229, 229))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

start_time = time.time()
core = ov.Core()
model_path = 'inception_v3.xml'
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name='MYRIAD')
model_load_time = (time.time() - start_time) * 1000
print(f"Model load time: {model_load_time} ms")

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

test_data_path = '/home/pi/Downloads/data/test'
labels = load_labels('/home/pi/Desktop/NCS2/leaf_labels.txt')
predictions = []
actual_labels = []
inference_times = []

for folder in os.listdir(test_data_path):
    folder_path = os.path.join(test_data_path, folder)
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    for image_file in image_files:
        image = preprocess_image(image_file)
        print("Model expected input shape:", model.input(0).get_shape())
        print("Input data shape for inference:", image.shape)
        first_inference_times = []
        for _ in range(10):
            start_time = time.time()
            result = compiled_model(inputs={input_key: image})[output_key]
            inference_time = (time.time() - start_time) * 1000
            first_inference_times.append(inference_time)
        inference_times.extend(first_inference_times)
        result_index = np.argmax(result)
        predictions.append(result_index)
        actual_labels.append(labels.index(folder))
        current_accuracy = accuracy_score(actual_labels, predictions)
        print(f"Image: {image_file}, Predicted label index: {result_index}")
        print(f"Current Accuracy: {current_accuracy * 100:.2f}%")

# Final overall accuracy and its standard deviation
final_accuracy = accuracy_score(actual_labels, predictions)
std_dev_accuracy = statistics.stdev([1 if a == p else 0 for a, p in zip(actual_labels, predictions)])
average_inference = statistics.mean(inference_times)
std_dev_inference = statistics.stdev(inference_times)

print(f"Average inference time: {average_inference} ms")
print(f"Standard deviation of inference times: {std_dev_inference} ms")
print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
print(f"Standard deviation of Accuracy: {std_dev_accuracy}")
