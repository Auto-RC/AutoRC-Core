#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")
if [ "$1" != "" ]; then
	raspivid -vf -o /home/pi/vid.h264 -n -t $1 -w 320 -h 180
	MP4Box -add vid.h264 /home/pi/Videos/vid_$DATE.mp4
	rm vid.h264
else
	raspivid -vf -o /home/pi/vid.h264 -n -t 999999 -w 320 -h 180
    MP4Box -add vid.h264 /home/pi/Videos/vid_$DATE.mp4
    rm vid.h264
fi
