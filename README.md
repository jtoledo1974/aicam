# aicam
Person detection from video stream, publishing to home assistant

python aicam.py --modeldir ../tflite1/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/ \
        --streamurl rtsp://user:password@192.168.1.241:554//h264Preview_01_sub \
        --resolution 640x480 --mqtt_host=localhost --mqtt_name se --threshold 0.65
         > se.log 2>&1

        & tail -f se.log
        | egrep --color=auto "INFO|h264"

