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
from pathlib import PurePath
import subprocess
import threading


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


class Mqtt:
    def __init__(self, host='localhost', port=1883, keepalive=60, bind_address="", name="aicam"):
        import paho.mqtt.client as mqtt
        self.client = client = mqtt.Client()

        # client.enable_logger()
        client.connect(host, port, keepalive, bind_address)
        client.loop_start()

        self.base = base = f"homeassistant/binary_sensor/motion_aicam_{name}"
        config_topic = f'{base}/config'
        config_msg = f'{{"name": "motion_aicam_{name}", "device_class": "motion", "json_attributes_topic": "{base}/attributes", "state_topic": "{base}/state"}}'
        client.publish(config_topic, config_msg)

        self.basecam = base = f"homeassistant/camera/aicam_{name}"
        config_topic = f'{base}/config'
        config_msg = f'{{"name": "aicam_{name}", "topic": "{base}"}}'

        logging.info(f"Publish {config_topic} {config_msg}")
        client.publish(config_topic, config_msg)

        self.state, self.confidence, self.fps, self.image = 'OFF', 0, 0, None

    def set_state(self, state, confidence, image=None, force=False):

        self.set_confidence(confidence)

        if (state == 'ON' or force) and image:
            self.set_image(image)

        if state != self.state or force:
            logging.info(f"Publish {self.base}/state {state}")
            self.client.publish(f"{self.base}/state", state)

        self.state = state

    def set_image(self, image):
        logging.info(f"Publish {self.basecam} 'img_data'")
        self.client.publish(self.basecam, image)

    def set_confidence(self, confidence):
        if confidence != self.confidence:
            logging.info(f"Publish {self.base}/attributes {{'confidence': {confidence}}}")
            self.client.publish(f"{self.base}/attributes", f'{{"confidence": {confidence}}}')
            self.confidence = confidence

    def set_fps(self, fps):
        self.set_state(self.state, self.confidence)
        topic, value = f"{self.base}/attributes", f'{{"fps": {fps}}}'
        logging.debug(f"Publish {topic} {value}")
        self.client.publish(topic, value)
        self.fps = fps

    def stop(self):
        pass
        # We avoid deleting the configuration entries so that we keep the last
        # known values when we stop the program
        # self.client.publish(f"{self.base}/config", "")
        self.client.publish(f"{self.basecam}/config", "")


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming"""

    def __init__(self, resolution=(640, 480)):
        # Initialize the camera image stream

        # Important for cameras that don't properly report UDP transport
        # Otherwise we get "Nonmatching transport in server reply" error
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.resolution = resolution
        self.lock = threading.Lock()
        self.connect(resolution)

    def connect(self, resolution):
        self.stream = cv2.VideoCapture(STREAM_URL)
        if not self.stream.isOpened():
            raise ConnectionError
        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        self.update_thread = Thread(target=self.update, args=())
        self.update_thread.start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                logging.debug("Releasing stream")
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            # In case of disconnect
            if not self.grabbed:
                logging.warning("Failed to grab frame. Possible disconnect")
                self.stream.release()
                self.connect(self.resolution)

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
        self.update_thread.join()


class Videorecorder:
    def __init__(self, port=5000, video_duration=20):
        self.port = port
        self.video_duration = video_duration
        self.recording_process = None

    def record_video(self, filename):
        if not self.recording_process:

            self.filename = filename

            # Create parent folder if nonexistent
            try:
                os.makedirs(PurePath(filename).parent)
            except FileExistsError:
                pass

            # Stream ts from server to file
            logging.info(f"Iniciando grabacion {filename}")
            cmd = f'/usr/bin/gst-launch-1.0 -v tcpclientsrc host=127.0.0.1 port={self.port} ! filesink location={filename}'
            self.recording_process = subprocess.Popen(cmd.split(" "))

            # Start time to cancel recording
            self.timer = threading.Timer(self.video_duration, self.stop_recording)
            self.timer.start()

        else:

            # Already recording. Delay end of recording
            logging.info("Retrasando fin de grabacion")
            self.timer.cancel()
            self.timer = threading.Timer(self.video_duration, self.stop_recording)
            self.timer.start()

    def stop_recording(self, *args, **kwargs):
        logging.info("Parando grabacion")
        self.recording_process.terminate()
        try:
            self.recording_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logging.error(("Did not terminate", self.recording_process))
        self.recording_process = None

        # Transcode to mp4
        old_fn = PurePath(self.filename)
        new_fn = old_fn.parent / (old_fn.stem + '.mp4')
        cmd = f"ffmpeg -i {old_fn} -c copy {new_fn} && rm {old_fn}"
        logging.info(cmd)
        logging.info(subprocess.run(cmd, shell=True))

    def terminate(self):
        if self.recording_process:
            self.timer.cancel()
            self.stop_recording({})


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
camera_name = args.mqtt_name
if args.mqtt_host != '':
    mqtt = Mqtt(args.mqtt_host, name=camera_name)

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

# Initialize minute FPM average
minute_frame_counter = 0
fpm_last_reset = datetime.now()


# Initialize video stream
logging.debug("Initializing VideoStream")
videostream = VideoStream(resolution=(imW, imH)).start()
time.sleep(1)

# Initialize video recorder
if camera_name == 'sw':
    port = 5000
elif camera_name == 'se':
    port = 5001
logging.debug("Initializing Videorecorder")
recorder = Videorecorder(port=port)
savedir = f"/srv/cameras/{camera_name}/video"

stop = False


def handler(signum, frame):
    global stop
    print(f"Dentro de handler stop {stop}")
    stop = True


signal.signal(signal.SIGINT, handler)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
first_frame = True
person_on_last = False
while not stop:
    logging.debug("Top of while 2")

    # time.sleep(1)
    # continue
    # logging.error("Dentro del while!!")

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    logging.debug("Before frame read")
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    try:
        frame = frame1.copy()
    except AttributeError:
        logging.warning("Failed to grab frame")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    logging.debug("Before detection")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    logging.debug("After invoke")
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Hold the highest confidence of person in image
    person_confidence = 0

    logging.debug("Before looping for detections")
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
            area = (xmax - xmin) * (ymax - ymin)
            x, y = xmax - xmin, ymax - ymin

            object_name = labels[int(classes[i])]
            # if object_name != 'person' or area < 2000 or area > 50000:
            # Activate only if a person is detected
            if object_name != 'person':
                continue
                # pass

            # Hack para evitar la detección de la mesa como persona por la noche
            false_positive = False
            if camera_name == 'se':
                if area > 86000:
                   false_positive = True
            elif camera_name == 'sw':
                if ((240 < x < 280) and (210 < y < 270) and (53000 < area < 73000)) or \
                   ((13 < x < 22) and (39 < y < 45) and (550 < area < 950)):  # Mesa y tronco, respectiv.
                    false_positive = True
            if false_positive:
                # timestring = datetime.now().strftime("%Y-%m/%d-%H.%M.%S")
                # cv2.imgwrite(f"/srv/camera/")
                continue
            logging.debug(f"X,Y = {x}, {y}; Area = {area};")

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

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Calcultate average FPS in a minute
    now = datetime.now()
    seconds_passed = (now - fpm_last_reset).seconds
    logging.debug((minute_frame_counter, now.second, seconds_passed))

    if now.second == 0 and seconds_passed > 1:
        fps_minute_average = minute_frame_counter / seconds_passed
        minute_frame_counter = 0
        fpm_last_reset = now
        logging.debug(f"-------------------- Average FPS {fps_minute_average:.2f}")

        if mqtt:
            mqtt.set_fps(fps_minute_average)
    minute_frame_counter += 1


    # Send results to home assistant
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

            person_on_last = True
        else:
            state = 'OFF'
            person_on_last = False
        mqtt.set_state(state, person_confidence, buf, force=force)

    # Record video
    if person_confidence > 0:
        timestring = datetime.now().strftime("%Y-%m/%d-%H.%M.%S")
        video_filename = f"{savedir}/{timestring}.ts"
        recorder.record_video(video_filename)


# Clean up
logging.debug("Cleaning up")
videostream.stop()
cv2.destroyAllWindows()
recorder.terminate()

if mqtt:
    mqtt.stop()

os._exit(0)
