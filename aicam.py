### Person detection off a video stream using TFLite COCO model ###
# Based off code from Evan Juras https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_stream.py

# Import packages
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import signal
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)


class Mqtt:
    def __init__(self, host='localhost', port=1883, keepalive=60, bind_address="", name="aicam"):
        import paho.mqtt.client as mqtt
        self.client = client = mqtt.Client()

        client.enable_logger()
        client.connect(host, port, keepalive, bind_address)
        client.loop_start()

        self.base = base = f"homeassistant/binary_sensor/{name}"
        config_topic = f'{base}/config'
        config_msg = f'{{"name": "{name}", "device_class": "motion", "json_attributes_topic": "{base}/attributes", "state_topic": "{base}/state"}}'
        client.publish(config_topic, config_msg)

        self.basecam = base = f"homeassistant/camera/aicam_{name}"
        config_topic = f'{base}/config'
        config_msg = f'{{"name": "aicam_{name}", "topic": "{base}"}}'

        logging.debug(f"Publish {config_topic} {config_msg}")
        client.publish(config_topic, config_msg)

        self.state, self.confidence = 'OFF', 0

    def set_state(self, state, confidence, image, force=False):

        if confidence != self.confidence:
            logging.debug(f"Publish {self.base}/attributes {{'confidence': {confidence}}}")
            self.client.publish(f"{self.base}/attributes", f'{{"confidence": {confidence}}}')

        if state != self.state or force:
            logging.debug(f"Publish {self.base}/state {state}")
            self.client.publish(f"{self.base}/state", state)

        if state == 'ON' or force:
            logging.debug(f"Publish {self.basecam} 'img_data'")
            self.client.publish(self.basecam, image)

        self.state, self.confidence = state, confidence

    def stop(self):
        self.client.publish(f"{self.base}/config", "")
        self.client.publish(f"{self.basecam}/config", "")


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming"""

    def __init__(self, resolution=(640, 480)):
        # Initialize the camera image stream

        # Important for cameras that don't properly report UDP transport
        # Otherwise we get "Nonmatching transport in server reply" error
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        self.stream = cv2.VideoCapture(STREAM_URL)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--streamurl', help='The full URL of the video stream e.g. http://ipaddress:port/stream/video.mjpeg',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')  # TODO The default resolution should be read from the streamed frame or the bounding boxes will be wrong!
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--mqtt_host', help="Address of mqtt server to which state is published", default='')
parser.add_argument('--mqtt_name', help="Name of binary_sensor device", default='aicam')

args = parser.parse_args()

mqtt = None
if args.mqtt_host != '':
    mqtt = Mqtt(args.mqtt_host, name=args.mqtt_name)

MODEL_NAME = args.modeldir
STREAM_URL = args.streamurl
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH)).start()
time.sleep(1)

stop = False


def handler(signum, frame):
    global stop
    print(f"Dentro de handler stop {stop}")
    stop = True


signal.signal(signal.SIGINT, handler)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
first_frame = True
while not stop:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Hold the highest confidence of person in image
    person_confidence = 0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            # print(imW, imH, boxes[1])
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            # if object_name != 'person' or area < 2000 or area > 50000:
            # Activate only if a person is detected
            if object_name != 'person':
                continue
                # pass

            if scores[i] > person_confidence:
                person_confidence = scores[i]

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                frame,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255), cv2.FILLED
            )  # Draw white box to put label text in
            cv2.putText(
                frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2
            )  # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    if mqtt:

        force = False
        # We provide the first image we get to home assistant
        # to verify that we are running properly
        if first_frame:
            retval, buf = cv2.imencode('.jpg', frame)
            buf = bytearray(buf)
            first_frame = False
            force = True
        else:
            buf = None

        if person_confidence > 0:
            state = 'ON'
            retval, buf = cv2.imencode('.jpg', frame)
            buf = bytearray(buf)
        else:
            state = 'OFF'
        mqtt.set_state(state, person_confidence, buf, force=force)

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    # if cv2.waitKey(1) == ord('q'):
    #    stop = True


# Clean up
cv2.destroyAllWindows()
videostream.stop()

if mqtt:
    mqtt.stop()
