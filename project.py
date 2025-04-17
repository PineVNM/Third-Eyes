"""
# Update the system:
sudo apt update && sudo apt upgrade -y  

----------------------------------------------------------------------

# Install Bluetooth Software
sudo apt install -y bluetooth bluez blueman

# Enable and Start Bluetooth Services
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

----------------------------------------------------------------------

# Install Python development tools:
sudo apt install python3-dev python3-pip -y

----------------------------------------------------------------------

# Install OpenCV:
sudo apt install libopencv-dev python3-opencv -y
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y build-essential cmake git libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base
sudo apt-get install  pkg-config libv4l-dev libxvidcore-dev libx264-dev gfortran openexr libatlas-base-dev libdc1394-dev


----------------------------------------------------------------------

# Install smbus2 library
sudo apt install python3-smbus2

----------------------------------------------------------------------

# Download YOLO Weights and Config File: 
mkdir -p ~/yolo
cd ~/yolo

# Download configuration and coco name for 608x608 version 
wget -P ~/yolo https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget -P ~/yolo https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget -P ~/yolo https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

# Read this to find more information about YOLOv4
https://github.com/AlexeyAB/darknet?tab=readme-ov-file

----------------------------------------------------------------------

# Install the audio
sudo apt-get install -y espeak

# Install pyttsx3 library
git clone https://github.com/nateshmbhat/pyttsx3.git
cd pyttsx3
python setup.py build
sudo python setup.py install

----------------------------------------------------------------------

# Ensuring the Script Runs Continuously:
sudo nano /etc/systemd/system/myservice.service

---------
[Unit]
Description=My Python Script Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/myscript.py
Restart=on-abort

[Install]
WantedBy=multi-user.target
---------

sudo systemctl daemon-reload
sudo systemctl enable myservice.service
sudo systemctl start myservice.service
sudo systemctl status myservice.service

----------------------------------------------------------------------

# Test if all the packeges is install
libcamera-hello                         # Test camera connection
sudo apt install cheese
cheese

python3 --version                       # Check if python is installed

-------

"""

import subprocess
import numpy as np
import cv2
import pyttsx3
from smbus2 import SMBus
from time import sleep, time
import os

# Directory setup for saving audio and images
os.makedirs("saved_audio", exist_ok=True)
os.makedirs("saved_images", exist_ok=True)

# Load the YOLO model and class names
net = cv2.dnn.readNet("yolo/yolov4.cfg", "yolo/yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("yolo/coco.names", "rt") as f:
    class_names = f.read().strip().split("\n")

# Initialize LIDAR-Lite v3 sensor parameters
I2C_ADDR = 0x62
bus = SMBus(1)

# Initialize text-to-speech engine
engine = pyttsx3.init()


# Function to read distance from LIDAR-Lite v3
def get_lidar_distance():
    try:
        bus.write_byte_data(I2C_ADDR, 0x00, 0x04)
        sleep(0.04)
        high = bus.read_byte_data(I2C_ADDR, 0x0f)
        low = bus.read_byte_data(I2C_ADDR, 0x10)
        distance_cm = (high << 8) + low
        return distance_cm
    except Exception as e:
        print(f"LIDAR Error: {e}")
        return None


# Convert distance to footsteps (1 step = 0.762 meters)
def estimate_footsteps(distance_cm):
    return round(distance_cm / 76.2)


# Function to determine direction of objects
def categorize_object_direction(frame_width, obj_x, obj_width):
    obj_center_x = obj_x + obj_width // 2
    left_boundary = frame_width * 0.4
    right_boundary = frame_width * 0.6

    if obj_center_x < left_boundary:
        return "left"
    elif obj_center_x > right_boundary:
        return "right"
    else:
        return "front"


# Function to detect objects and measure distances
def get_object_distances(frame, lidar_distance):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    detected_objects = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, width, height = box.astype("int")
                direction = categorize_object_direction(frame.shape[1], x, width)
                detected_objects.append((class_names[class_id], confidence, x, y, width, height, lidar_distance, direction))

    return detected_objects

def report_obstacles(detected_objects):
    for index, (obj_name, confidence, _, _, _, _, distance_cm, direction) in enumerate(detected_objects):
        steps = estimate_footsteps(distance_cm)
        message = f"{obj_name} is {steps} steps to the {direction}."
        engine.say(message)
        print(message)
        audio_file = f"saved_audio/obstacle_{index + 1}.mp3"
        engine.save_to_file(message, audio_file)
    engine.runAndWait()

def read_mjpeg_frame(process):
    buffer = bytearray()
    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                # No more data, end of file/stream
                break
            buffer.extend(chunk)
            # Attempt to find the start (SOI) and end (EOI) markers of a JPEG frame
            start = buffer.find(b'\xff\xd8')  # Start of Image (SOI) marker
            end = buffer.find(b'\xff\xd9', start)  # End of Image (EOI) marker
            
            if start != -1 and end != -1:
                frame_data = buffer[start:end + 2]
                buffer = buffer[end + 2:]  # Clear the buffer up to the end of the processed frame
                return np.frombuffer(frame_data, dtype=np.uint8)
            
    except Exception as e:
        print(f"Error reading frame data: {e}")
    
    return None


def main():
    width, height = 640, 480
    timeout_seconds = 2  # The desired timeout in seconds
    timeout_milliseconds = timeout_seconds * 1000  # Convert seconds to milliseconds
    command = ['libcamera-vid', '-t', str(timeout_milliseconds), '--inline', '--codec', 'mjpeg', '-o', '-', '-n', '--width', str(width), '--height', str(height)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    while True:
        frame_data = read_mjpeg_frame(process)
        if frame_data is not None:
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode frame, skipping...")
                continue

            lidar_distance = get_lidar_distance()
            if lidar_distance is None:
                continue

            detected_objects = get_object_distances(frame, lidar_distance)
            report_obstacles(detected_objects)
            for obj_name, confidence, x, y, width, height, direction, lidar_distance in detected_objects:
                # Now correctly handling all eight items
                #print(f"{obj_name} with {confidence*100:.2f}% confidence is {lidar_distance}cm away to the {direction}")
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name} {confidence*100:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            image_filename = f"saved_images/image_{int(time())}.jpg"
            cv2.imwrite(image_filename, frame)
            cv2.imshow('Object Detection', frame)
        else:
            print("No complete frame was found or end of stream.")
            break

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        #sleep(5)  # Delay for continuous updates

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


