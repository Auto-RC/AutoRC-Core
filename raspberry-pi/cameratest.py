import os
import time
import numpy as np
from PIL import Image
import glob
from threading import Thread

class BaseCamera:

    def run_threaded(self):
        print(len(self.frame), len(self.frame[0]))
        return self.frame


class PiCamera(BaseCamera):
    def __init__(self, resolution=(320, 180), framerate=20):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        resolution = (resolution[1], resolution[0])
        # initialize the camera and stream
        self.camera = PiCamera()  # PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="rgb",
                                                     use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('PiCamera loaded.. .warming camera')
        time.sleep(2)

    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
        print("a", frame)
        return frame

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stopping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()

cam = PiCamera()
t = Thread(target=cam.update, args=())
t.daemon = True
t.start()
time.sleep(1)
print("updated")
t = time.time()
i = 0
while time.time() - t < 1:
    s = time.time()
    cam.run_threaded()
    print(time.time() - s, i)
    i += 1
print("time taken: ", time.time() - t)
cam.shutdown()
