import os
import pathlib
import time
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from socket import *

def measure_inference_time(interpreter, image):
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return (end_time - start_time) 

def preprocess_image(image, size):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image

def getCurrentAddress():
    s = socket(AF_INET, SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('8.8.8.8', 1))
        address = s.getsockname()[0]
    except Exception:
        address = '127.0.0.1'
    finally:
        s.close()
    return address

# Model setup
script_dir = pathlib.Path("./")
model_file = os.path.join(script_dir, 'inception_v3_224_quant_float32.tflite')

start_load_time = time.time()
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
end_load_time = time.time()
model_load_time = (end_load_time - start_load_time)
input_size = common.input_size(interpreter)

# Setup UDP server
address = str(getCurrentAddress())
s = socket(AF_INET, SOCK_DGRAM)
server_address = (address, 9999)
s.bind(server_address)
s.setblocking(0)
print("Server has started")

frames_per_second = []
frame_counter = 0
prev_time = time.time()

while frame_counter < 3351:
    try:
        data, _ = s.recvfrom(65535)  
        if data:
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Display the received image
            #cv2.imshow('Received Image', image)

            processed_image = preprocess_image(image, input_size)
            common.set_input(interpreter, processed_image)

            # Measure inference time
            inference_time = measure_inference_time(interpreter, processed_image)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            frames_per_second.append(fps)
            print("Current FPS: {:.2f}".format(fps))

            frame_counter += 1

    except BlockingIOError:
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
