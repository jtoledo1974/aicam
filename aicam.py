"""Person detection off a video stream using TFLite COCO model."""
# Based off code from Evan Juras https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_stream.py

# Import packages
import argparse
import importlib.util
import logging
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

import cv2
import numpy as np

# ruff: noqa: ANN001, ANN201, D102, DTZ005, FBT002, PLR2004

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Print the command line used
logger.debug(" ".join(["python3", *os.sys.argv]))


@dataclass
class MqttConnection:
    """MQTT connection parameters."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    bind_address: str = ""
    name: str = "aicam"


class Mqtt:
    """MQTT client wrapper."""

    def __init__(self, conn_data: MqttConnection):
        import paho.mqtt.client as mqtt

        host, port, keepalive, bind_address, name = (
            conn_data.host,
            conn_data.port,
            conn_data.keepalive,
            conn_data.bind_address,
            conn_data.name,
        )

        self.client = client = mqtt.Client()

        client.connect(host, port, keepalive, bind_address)
        client.loop_start()

        self.base = base = f"homeassistant/binary_sensor/motion_aicam_{name}"
        config_topic = f"{base}/config"
        config_msg = f'{{"name": "motion_aicam_{name}", "device_class": "motion", "json_attributes_topic": "{base}/attributes", "state_topic": "{base}/state"}}'
        client.publish(config_topic, config_msg)

        self.basecam = base = f"homeassistant/camera/aicam_{name}"
        config_topic = f"{base}/config"
        config_msg = f'{{"name": "aicam_{name}", "topic": "{base}"}}'

        logger.info("Publish %s %s", config_topic, config_msg)
        client.publish(config_topic, config_msg)

        self.state, self.confidence, self.fps, self.image = "OFF", 0, 0, None

    def set_state(self, state, confidence, image=None, force=False):
        self.set_confidence(confidence)

        if (state == "ON" or force) and image:
            self.set_image(image)

        if state != self.state or force:
            logger.info("Publish %s %s", f"{self.base}/state", state)
            self.client.publish(f"{self.base}/state", state)

        self.state = state

    def set_image(self, image):
        logger.info("Publish %s 'img_data'", self.basecam)
        self.client.publish(self.basecam, image)

    def set_confidence(self, confidence):
        if confidence != self.confidence:
            logger.info(
                "Publish %s {'confidence': %s}", f"{self.base}/attributes", confidence
            )
            self.client.publish(
                f"{self.base}/attributes", f'{{"confidence": {confidence}}}'
            )
            self.confidence = confidence

    def set_fps(self, fps):
        self.set_state(self.state, self.confidence)
        topic, value = f"{self.base}/attributes", f'{{"fps": {fps}}}'
        logger.debug("Publish %s %s", topic, value)
        self.client.publish(topic, value)
        self.fps = fps

    def stop(self):
        # We avoid deleting the configuration entries so that we keep the last
        # known values when we stop the program
        self.client.publish(f"{self.basecam}/config", "")


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming."""

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
                logger.debug("Releasing stream")
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            # In case of disconnect
            if not self.grabbed:
                logger.warning("Failed to grab frame. Possible disconnect")
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
    """Class to record video from a stream."""

    def __init__(
        self, url: str, latency: int = 10000, port: int = 5000, video_duration=20
    ):
        self.url = url
        self.latency = latency
        self.port = port
        self.video_duration = video_duration
        self.recording_process: Optional[subprocess.Popen[bytes]] = None

        self.launch_delayed_video_server()

    def launch_delayed_video_server(self):
        """Launch a server that streams a delayed version of the given camera URL."""
        logger.info("Launching delayed video server")
        cmd = [
            "/usr/bin/gst-launch-1.0",
            "rtspsrc",
            f"latency={self.latency}",
            f"location={self.url}",
            "name=rtspsrc",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "config_interval=-1",
            "!",
            "mpegtsmux",
            "name=mux",
            "!",
            "tcpserversink",
            "host=127.0.0.1",
            f"port={self.port}",
            "rtspsrc.",
            "!",
            "rtpmp4gdepay",
            "!",
            "aacparse",
            "!",
            "mux.",
        ]

        logger.info(" ".join(cmd))
        self.delayed_video_server = subprocess.Popen(cmd)

    def record_video(self, filename):
        if not self.recording_process:
            self.filename = filename

            # Create parent folder if nonexistent
            with suppress(FileExistsError):
                Path(filename).parent.mkdir(parents=True)

            # Stream ts from server to file
            logger.info("Starting recording %s", filename)
            cmd = f"/usr/bin/gst-launch-1.0 -v tcpclientsrc host=127.0.0.1 port={self.port} ! filesink location={filename}"
            self.recording_process = subprocess.Popen(cmd.split(" "))

            # Start time to cancel recording
            self.timer = threading.Timer(self.video_duration, self.stop_recording)
            self.timer.start()

        else:
            # Already recording. Delay end of recording
            logger.info("Retrasando fin de grabacion")
            self.timer.cancel()
            self.timer = threading.Timer(self.video_duration, self.stop_recording)
            self.timer.start()

    def stop_recording(self, *_args, **_kwargs):
        logger.info("Parando grabacion")
        self.recording_process.terminate()
        try:
            self.recording_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logger.exception(("Did not terminate", self.recording_process))
        self.recording_process = None

        # Transcode to mp4
        old_fn = Path(self.filename)
        new_fn = old_fn.parent / (old_fn.stem + ".mp4")
        cmd = f"ffmpeg -i {old_fn} -c copy {new_fn} && rm {old_fn}"
        logger.info(cmd)
        logger.info(subprocess.run(cmd, shell=True))

    def terminate(self):
        if self.recording_process:
            self.timer.cancel()
            self.stop_recording({})


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--streamurl",
    help="The full URL of the video stream e.g. http://ipaddress:port/stream/video.mjpeg",
    required=True,
)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
    default="640x480",
)  # TODO The default resolution should be read from the streamed frame or the bounding boxes will be wrong!
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
)
parser.add_argument(
    "--mqtt_host", help="Address of mqtt server to which state is published", default=""
)
parser.add_argument("--mqtt_name", help="Name of binary_sensor device", default="aicam")
parser.add_argument("--max_fps", help="Maximum FPS", default=3)

args = parser.parse_args()

mqtt = None
camera_name = args.mqtt_name
if args.mqtt_host != "":
    conn_data = MqttConnection(args.mqtt_host, name=camera_name)
    mqtt = Mqtt(conn_data)

MODEL_NAME: str = args.modeldir
STREAM_URL = args.streamurl
GRAPH_NAME: str = args.graph
LABELMAP_NAME: str = args.labels
min_conf_threshold = float(args.threshold)
res_width, res_height = args.resolution.split("x")
img_width, img_height = int(res_width), int(res_height)
use_tpu = args.edgetpu
max_fps = float(args.max_fps)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec("tflite_runtime")
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_tpu:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_tpu:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_tpu and GRAPH_NAME == "detect.tflite":
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    GRAPH_NAME = "edgetpu.tflite"

# Get path to current working directory
CWD_PATH = Path.cwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = CWD_PATH / MODEL_NAME / GRAPH_NAME

# Path to label map file
PATH_TO_LABELS = CWD_PATH / MODEL_NAME / LABELMAP_NAME

# Load the label map
with PATH_TO_LABELS.open() as f:
    labels = [line.strip() for line in f.readlines()]

# Labels are officially listed in https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt
# But the format expected by this script is https://raw.githubusercontent.com/JerryKurata/TFlite-object-detection/main/labelmap.txt
# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == "???":
    del labels[0]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_tpu:
    interpreter = Interpreter(
        model_path=str(PATH_TO_CKPT),
        experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
    )
    logger.info(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=str(PATH_TO_CKPT))

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = input_details[0]["dtype"] == np.float32

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize minute FPM average
minute_frame_counter = 0
fpm_last_reset = datetime.now()


# Initialize video stream
logger.debug("Initializing VideoStream")
videostream = VideoStream(resolution=(img_width, img_height)).start()
time.sleep(1)

# Initialize video recorder
if camera_name == "sw":
    port = 5000
elif camera_name == "se":
    port = 5001
logger.debug("Initializing Videorecorder")
recorder = Videorecorder(url=STREAM_URL, port=port)
savedir = f"/recordings/{camera_name}"

stop = False


def handler(_signum, _frame):
    """Handle SIGINT and SIGTERM."""
    global stop
    logger.debug("Dentro de handler stop %s", stop)
    stop = True


signal.signal(signal.SIGINT, handler)

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
first_frame = True
person_on_last = False

while not stop:
    logger.debug("Top of while 2")

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    logger.debug("Before frame read")
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    try:
        frame = frame1.copy()
    except AttributeError:
        logger.warning("Failed to grab frame")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    logger.debug("Before detection")
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    logger.debug("After invoke")
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]["index"])[
        0
    ]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]["index"])[
        0
    ]  # Confidence of detected objects

    # Hold the highest confidence of person in image
    person_confidence = 0

    logger.debug("Before looping for detections")
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * img_height)))
            xmin = int(max(1, (boxes[i][1] * img_width)))
            ymax = int(min(img_height, (boxes[i][2] * img_height)))
            xmax = int(min(img_width, (boxes[i][3] * img_width)))
            area = (xmax - xmin) * (ymax - ymin)
            x, y = xmax - xmin, ymax - ymin

            object_name = labels[int(classes[i])]
            # if object_name != 'person' or area < 2000 or area > 50000:
            # Activate only if a person is detected
            if object_name != "person":
                continue
                # pass

            # Hack para evitar la detección de la mesa como persona por la noche
            false_positive = False
            if camera_name == "se":
                if area > 86000:
                    false_positive = True
            elif camera_name == "sw" and (
                ((240 < x < 280) and (210 < y < 270) and (53000 < area < 73000))
                or ((13 < x < 22) and (39 < y < 45) and (550 < area < 950))
            ):  # Mesa y tronco, respectiv.
                false_positive = True
            if false_positive:
                continue
            logger.debug("X,Y = %s, %s; Area = %s;", x, y, area)

            if scores[i] > person_confidence:
                person_confidence = scores[i]

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            label = "%s: %d%%" % (
                object_name,
                int(scores[i] * 100),
            )  # Example: 'person: 72%'
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            label_ymin = max(
                ymin, label_size[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                frame,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0], label_ymin + base_line - 10),
                (255, 255, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )  # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(
        frame,
        f"FPS: {frame_rate_calc:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # All the results have been drawn on the frame, so it's time to display it.

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Sleep so that we keep a maximum of max_fps
    if frame_rate_calc >= max_fps:
        time.sleep(1 / max_fps - time1)

    # Calcultate average FPS in a minute
    now = datetime.now()
    seconds_passed = (now - fpm_last_reset).seconds
    logger.debug((minute_frame_counter, now.second, seconds_passed))

    if now.second == 0 and seconds_passed > 1:
        fps_minute_average = minute_frame_counter / seconds_passed
        minute_frame_counter = 0
        fpm_last_reset = now
        logger.debug("-------------------- Average FPS %.2f", fps_minute_average)

        if mqtt:
            mqtt.set_fps(fps_minute_average)
    minute_frame_counter += 1

    # Send results to home assistant
    if mqtt:
        force = False
        # We provide the first image we get to home assistant
        # to verify that we are running properly
        if first_frame:
            retval, buf = cv2.imencode(".jpg", frame)
            buf = bytearray(buf)
            first_frame = False
            force = True
        else:
            buf = None

        if person_confidence > 0:
            state = "ON"
            retval, buf = cv2.imencode(".jpg", frame)
            buf = bytearray(buf)

            person_on_last = True
        else:
            state = "OFF"
            person_on_last = False
        mqtt.set_state(state, person_confidence, buf, force=force)

    # Record video
    if person_confidence > 0:
        timestring = datetime.now().strftime("%Y-%m/%d-%H.%M.%S")
        video_filename = f"{savedir}/{timestring}.ts"
        recorder.record_video(video_filename)


# Clean up
logger.debug("Cleaning up")
videostream.stop()
cv2.destroyAllWindows()
recorder.terminate()

if mqtt:
    mqtt.stop()

os._exit(0)  # Make sure all threads exit.
