import time
import picamera

with picamera.PiCamera() as camera:

    camera.resolution = (1024, 768)
    camera.start_preview()
    # Camera warm-up time
    time.sleep(2)
    for i in range(200):
        camera.capture('track1/pic{}.jpg'.format(i))

