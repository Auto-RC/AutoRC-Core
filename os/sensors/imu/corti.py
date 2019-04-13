# ==================================================================================================
#                                          GLOBAL IMPORTS
# ==================================================================================================

import board
import busio
import adafruit_lsm9ds1
import threading
import time
import logging

# ==================================================================================================
#                                            LOGGER SETUP
# ==================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ==================================================================================================
#                                              CORTI
# ==================================================================================================

class Corti(threading.Thread):

    def __init__(self, update_interval_ms=50):

        # Thread parameters
        self.thread_name = "Drive"
        threading.Thread.__init__(self, name=self.thread_name)
        self._stop_event = threading.Event()

        self.enabled = False
        self.update_interval_ms = update_interval_ms

        self.conn = busio.I2C(board.SCL,board.SDA)
        self.imu = adafruit_lsm9ds1.LSM9DS1_I2C(self.conn)

        self.acceleration = 0

    def run(self):

        logger.info("Corti enabled...")

        self.enabled = True
        while self.enabled == True:

            self.acceleration = round(list(self.imu.acceleration)[0],3)

            logger.debug("Acceleration: {}g".format(self.acceleration))

            time.sleep(self.update_interval_ms/1000)

        logger.info("Corti disabled.")

    def disable(self):

        self.enabled = False
        self._stop_event.set()

    def stopped(self):

        return self._stop_event.is_set()

# ==================================================================================================
#                                           UNIT TEST
# ==================================================================================================

if __name__ == '__main__':

    corti = Corti(update_interval_ms=10)
    corti.enabled = True
    corti.run()

