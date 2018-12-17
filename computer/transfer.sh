#!/bin/bash

echo "transfering photos"
rm -rf photos/img*
scp -r pi@10.0.0.239:/home/pi/camera/photos /Users/arnavgupta/RCcar
