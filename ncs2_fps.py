import numpy as np
import cv2
from socket import *
import time
import openvino.runtime as ov
import statistics

def normalize(image):
    return image / 255.0

def preprocess_image(image):
    image = cv2.resize(image, (229, 229))
    image = normalize(image)  
    #image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

def getCurrentAddress():
    s = socket(AF_INET, SOCK_DGRAM) 
    s.settimeout(0)
    try:
        s.connect(('8.8.8.8', 1))
        address = s.getsockname()[0]
        print(address)
    except Exception:
        address = '127.0.0.1'
    finally:
        s.close()
    return address

labels = load_labels('/home/pi/Desktop/NCS2/leaf_labels.txt')
predictions = []
inference_times = []
frames_per_second = []

start_time = time.time()
core = ov.Core()
model = core.read_model(model='/home/pi/Desktop/NCS2/inception_v3.xml')
compiled_model = core.compile_model(model=model, device_name='MYRIAD')
model_load_time = time.time() - start_time
print(f"Model load time: {model_load_time} seconds")

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

address = str(getCurrentAddress())
s = socket(AF_INET, SOCK_DGRAM) 
server_address = (address, 9999) 
s.bind(server_address)
s.setblocking(0)
print("Server has started")

frame_counter = 0
prev_time = time.time()

while frame_counter < 3351:
    try:
        data, _ = s.recvfrom(384*288*4)
        receive_data = np.frombuffer(data, dtype='uint8')
        r_img = cv2.imdecode(receive_data, 1)
        r_img = cv2.rotate(r_img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('server', r_img)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frames_per_second.append(fps)

        inference_start_time = time.time()
        image = preprocess_image(r_img)
        result = compiled_model(inputs={input_key: image})[output_key]
        inference_end_time = time.time()
        inference_time_ms = (inference_end_time - inference_start_time) * 1000
        inference_times.append(inference_time_ms)

        result_index = np.argmax(result)
        predictions.append(result_index)

        frame_counter += 1

        print(f"Frame {frame_counter}, Inference Time: {inference_time_ms:.2f} ms")

    except BlockingIOError as e:
        pass
    except KeyboardInterrupt:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

s.close()
cv2.destroyAllWindows()

if frames_per_second:
    fps_mean = np.mean(frames_per_second)
    fps_std = np.std(frames_per_second)
    print("Average FPS: {:.2f}".format(fps_mean))
    print("Standard Deviation of FPS: {:.2f}".format(fps_std))
else:
    print("No frames were processed.")
