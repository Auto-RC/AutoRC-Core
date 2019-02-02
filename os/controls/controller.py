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

utility_dir = parent_dir + r'/utility'

sys.path.append(utility_dir)

from logger import *


# ==================================================================================================
#                                          CONTROLLER
# ==================================================================================================

#TODO: Update to non-pygame controller class

class Controller(threading.Thread):

    def __init__(self, wait_interval_ms):

        logger.info("Initializing controller thread...")

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Controller"
        threading.Thread.__init__(self, name=self.thread_name)

        logger.debug("Thread variables for controller set")

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.wait_interval_ms = wait_interval_ms
        self.stop_flag = False

        # Initializing the controller index
        # ------------------------------------------------------------------------------------------
        self.ctrl_axis_index = dict()
        self.ctrl_axis_index['r_j_x'] = 0  # right joystick x
        self.ctrl_axis_index['r_j_y'] = 1  # right joystick y
        self.ctrl_axis_index['l_j_x'] = 2  # left joystick x
        self.ctrl_axis_index['l_j_y'] = 5  # left joystick y
        self.ctrl_axis_index['a_y'] =   9  # arrow y
        self.ctrl_axis_index['a_x'] =   10  # arrow x

        self.ctrl_btn_index = dict()
        self.ctrl_btn_index['[]'] =     0  # If you do not see a square, you are sad.
        self.ctrl_btn_index['x'] =      1
        self.ctrl_btn_index['O'] =      2
        self.ctrl_btn_index['^'] =      3
        self.ctrl_btn_index['l_b'] =    4  # left bumper
        self.ctrl_btn_index['r_b'] =    5  # right bumper
        self.ctrl_btn_index['l_t'] =    6  # left trigger
        self.ctrl_btn_index['r_t'] =    7  # right trigger
        self.ctrl_btn_index['pwr'] =    12 # power

        logger.debug("Controller dictionaries intialized")


        # Initializing the dict which store controller values
        # ------------------------------------------------------------------------------------------
        self.ctrl_axis_val = dict()
        for key , value in self.ctrl_axis_index.items():
            self.ctrl_axis_val[key] = 0.0

        self.ctrl_btn_val = dict()
        for key , value in self.ctrl_btn_index.items():
            self.ctrl_btn_val[key] = False

        # Initializing PyGame
        # ------------------------------------------------------------------------------------------
        pygame.init()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        screen = pygame.display.set_mode((1, 1))

        logger.info("Done initializing controller thread")

    def update(self):

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        for event in pygame.event.get():
            pass

        for key , value in self.ctrl_axis_index.items():
            self.ctrl_axis_val[key] = joystick.get_axis(value)

        # logger.debug(self.ctrl_axis_val)

        for key , value in self.ctrl_btn_index.items():
            self.ctrl_btn_val[key] = joystick.get_button(value)

        # logger.debug(self.ctrl_btn_val)

        if self.ctrl_btn_val['pwr']:
            self.stop()

    def run(self):

        logger.info("Controller thread started")

        while self.stop_flag == False:

            self.update()
            time.sleep(self.wait_interval_ms/1000.0)

        logger.info("Controller thread stopped")

    def stop(self):

        self.stop_flag = True


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    c = Controller(wait_interval_ms=10)

    c.run()
