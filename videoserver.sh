#!/bin/bash
URL=$1
LATENCY=$2
PORT=$3
HOST=127.0.0.1

if [ $# -eq 3 ]; then
	COMMAND="gst-launch-1.0 rtspsrc latency=$LATENCY location=$URL name=rtspsrc ! rtph264depay ! h264parse config_interval=-1 ! mpegtsmux name=mux ! tcpserversink host=$HOST port=$PORT rtspsrc. ! rtpmp4gdepay ! aacparse ! mux."
elif [ $# -eq 4 ]; then
	COMMAND="gst-launch-1.0 rtspsrc latency=$LATENCY location=$URL name=rtspsrc ! rtph264depay ! h264parse config_interval=-1 ! mpegtsmux name=mux ! tcpserversink host=$HOST port=$PORT"
else
	echo "videoserver.sh url latency port [noaudio]"
	exit 1
fi	

echo $COMMAND
$COMMAND
