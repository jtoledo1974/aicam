FROM python:3.8

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime paho-mqtt opencv-python==3.4.11.41


RUN git clone https://github.com/jtoledo1974/aicam.git

RUN mkdir /models
WORKDIR /models
RUN wget http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz && \
    tar -xvf ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz && \
    rm ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz

WORKDIR /models/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19
RUN wget https://raw.githubusercontent.com/JerryKurata/TFlite-object-detection/main/labelmap.txt
RUN ln model.tflite detect.tflite

WORKDIR /aicam

CMD ["python", "aicam.py", "--modeldir", "${MODELDIR}", "--streamurl", "${STREAMURL}"]
