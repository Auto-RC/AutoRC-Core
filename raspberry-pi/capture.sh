#!/bin/bash

rm -rf /home/pi/Pictures/img*
raspivid -w 320 -h 180 -n -t $1 -vf -b 2000000 -fps 40 -o - | gst-launch-1.0 fdsrc ! video/x-h264,framerate=40/1,stream-format=byte-stream ! decodebin ! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! jpegenc ! multifilesink location=/home/pi/Pictures/img_%04d.jpg

