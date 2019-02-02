import time


class PiCamera:

    def __init__(self, resolution=, framerate, format):

        from picamera.array import PiRGBArray
        from picamera import PiCamera
        resolution = (resolution[1], resolution[0])
        # initialize the camera and stream
        self.camera = PiCamera()  # PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.format=format
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format=self.format,
                                                     use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('PiCamera loaded ... warming camera')
        time.sleep(1)

    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
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

    def run_threaded(self):
        return self.frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stopping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()