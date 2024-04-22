import os
import pathlib
import time
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

# Function to measure inference time
def measure_inference_time(interpreter, image):
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return (end_time - start_time) * 1000.0  # Convert to milliseconds

script_dir = pathlib.Path("./")
model_file = os.path.join(script_dir, 'inception_v3_224_quant_float32_edgetpu.tflite')
label_file = os.path.join(script_dir, 'leaf_labels.txt')

test_data_path = '/home/pi/Downloads/data/test'

# Measure the time to load the model into memory
start_load_time = time.time()
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
end_load_time = time.time()
model_load_time = (end_load_time - start_load_time) * 1000.0

image_files = []
for root, dirs, files in os.walk(test_data_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(root, file))

inference_times = []
total_accuracy = 0.0

# Repeat first inference 10 times and calculate its average time
first_image = cv2.imread(image_files[0])
first_image = cv2.resize(first_image, common.input_size(interpreter))
first_image = first_image.astype(np.float32) / 255.0
common.set_input(interpreter, first_image)

first_inference_times = [measure_inference_time(interpreter, first_image) for _ in range(10)]
average_first_inference_time = np.mean(first_inference_times)

for image_file in image_files:
    # Load and prepare the image
    image = cv2.imread(image_file)
    image = cv2.resize(image, common.input_size(interpreter))
    image = image.astype(np.float32) / 255.0
    common.set_input(interpreter, image)

    inference_time = measure_inference_time(interpreter, image)
    inference_times.append(inference_time)

    classes = classify.get_classes(interpreter, top_k=1)
    labels = dataset.read_label_file(label_file)

    ground_truth = os.path.basename(os.path.dirname(image_file))

    top_class = classes[0].id
    if labels.get(top_class, top_class) == ground_truth:
        total_accuracy += 1


average_inference_time = np.mean(inference_times)
std_deviation_inference_time = np.std(inference_times)
average_accuracy = (total_accuracy / len(image_files)) * 100

print('-------RESULTS--------')
print('Model Load Time: %.2fms' % model_load_time)
print('Average First Inference Time (10 runs): %.2fms' % average_first_inference_time)
print('Average Inference Time: %.2fms' % average_inference_time)
print('Standard Deviation of Inference Time: %.2fms' % std_deviation_inference_time)
print('Average Accuracy: %.2f%%' % average_accuracy)
