import time
import logging
import threading
from autorc.vehicle.vision.retina import Retina

class CortexBasic(threading.Thread):

    def __init__(self, update_interval_ms, oculus, corti, controller):

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        self.oculus = oculus
        self.retina = Retina()

        self.retina.fil_hsv_l[2] = 180
        self.retina.fil_hsv_u[1] = 100

        self.thread_name = "Cortex"
        threading.Thread.__init__(self, name=self.thread_name)

        self.enabled = False
        self.update_interval_ms = update_interval_ms

        self.angles = [None, None, None]
        self.midpoints = [None, None, None]

    def process_frame(self):

        # Setting the current frame
        self.retina.frame = self.oculus.get_frame()

        # Detecting lines
        if self.retina.frame is not None:
            self.angles , self.midpoints = self.retina.process()


    def enable(self):

        self.enabled = True

    def disable(self):

        self.enabled = False

    def run(self):

        while True:

            if self.enabled == True:
                self.process_frame()

            time.sleep(self.update_interval_ms / 1000)

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()

    retina.load_npy(file_name='/Users/arnavgupta/car_data/raw_npy/oculus-2019-06-16 20;49;28.264824.npy')
    retina.test_line_detection()





