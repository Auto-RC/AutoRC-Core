#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

raspistill -vf -n -t 500 -o /home/pi/camera/photos/$DATE.jpg
