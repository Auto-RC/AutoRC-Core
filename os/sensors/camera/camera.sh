#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")

raspistill -n -t 500 -o /home/pi/Pictures/$DATE.jpg
