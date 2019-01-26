# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import threading
import pygame

# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
utility_dir = parent_dir + r'\utility'
print(utility_dir)

sys.path.append(utility_dir)

from logger import *


# ==================================================================================================
#                                          CONTROLLER
# ==================================================================================================

#TODO: Update to non-pygame controller class

class Controller(threading.Thread):

    def __init__(self, wait_interval):

        logger.setLevel(logging.INFO)

        logger.info("Initializing class")

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Controller"
        threading.Thread.__init__(self, name=self.thread_name)
        self.stop_flag = False

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.wait_interval = wait_interval

        # Initializing the controller input index
        # ------------------------------------------------------------------------------------------
        self.axes = [3, 4, 2]
        self.buttons = [1, 12]
        self.throttle = -1
        self.brake = -1
        self.steering = 0
        self.capturing = False

        # Initializing PyGame
        # ------------------------------------------------------------------------------------------
        pygame.init()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        screen = pygame.display.set_mode((1, 1))

    def update(self):

        for event in pygame.event.get():
            pass

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        vals = [0] * len(self.axes)

        for i in range(len(self.axes)):
            vals[i] = joystick.get_axis(self.axes[i])

        self.throttle = vals[1]
        self.brake = vals[0]
        self.steering = vals[2]

        vals = [0] * len(self.buttons)

        for i in range(len(self.buttons)):
            vals[i] = joystick.get_button(self.buttons[i])

        if vals[0]:
            logger.debug("x pressed")
            print("x pressed")
            self.capturing = not self.capturing
            if self.capturing:
                print("capturing")
            else:
                print("saving")

        if vals[1]:
            self.on = not self.on

    def run(self):

        logger.info("Controller thread started")

        while self.stop_flag == False:

            self.update()
            time.sleep(self.wait_interval)

        logger.info("Controller thread input")

    def stop(self):

        self.stop_flag = True


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    c = Controller(0.5)

    c.run()
