#!/bin/bash

rm -rf /home/pi/Pictures/img*
if [ "$2" != " " ]; then
        raspivid -w 320 -h 180 -n -t $1 -b 2000000 -fps 40 -o - | gst-launch-1.0 fdsrc ! video/x-h264,framerate=40/1,s$
else
        raspivid -w 320 -h 180 -n -t $1 -vf -b 2000000 -fps 40 -o - | gst-launch-1.0 fdsrc ! video/x-h264,framerate=40$
fi

