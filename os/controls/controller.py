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
print(utility_dir)

sys.path.append(utility_dir)

from logger import *


# ==================================================================================================
#                                          CONTROLLER
# ==================================================================================================

#TODO: Update to non-pygame controller class

class Controller(threading.Thread):

    def __init__(self, wait_interval_ms):

        logger.setLevel(logging.DEBUG)

        logger.info("Initializing controller thread...")

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Controller"
        threading.Thread.__init__(self, name=self.thread_name)

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.wait_interval_ms = wait_interval_ms
        self.stop_flag = False

        # Initializing the controller index
        # ------------------------------------------------------------------------------------------
        self.ctrl_axis_index = dict()
        self.ctrl_axis_index['r_joystick_x'] = 0
        self.ctrl_axis_index['r_joystick_y'] = 1
        self.ctrl_axis_index['l_joystick_x'] = 2
        self.ctrl_axis_index['l_joystick_y'] = 5
        self.ctrl_axis_index['lr_arrow'] =     9
        self.ctrl_axis_index['ud_arrow'] =     10

        self.ctrl_btn_index = dict()
        self.ctrl_btn_index['[]_btn'] =        0  # If you do not see a square, you are sad.
        self.ctrl_btn_index['x_btn'] =         1
        self.ctrl_btn_index['O_btn'] =         2
        self.ctrl_btn_index['^_btn'] =         3
        self.ctrl_btn_index['left_bumper'] =   4
        self.ctrl_btn_index['right_bumper'] =  5
        self.ctrl_btn_index['left_trigger'] =  6
        self.ctrl_btn_index['right_trigger'] = 7
        self.ctrl_btn_index['pwr'] =           12


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

        for event in pygame.event.get():
            pass

        logger.info("Done initializing controller thread")

    def update(self):

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        for key , value in self.ctrl_axis_index.items():
            self.ctrl_axis_val[key] = self.joystick.get_axis(value)
            logger.debug("{}[{}] = {}".format(key,value,self.ctrl_axis_val[key]))

        # logger.debug(self.ctrl_axis_val)

        for key , value in self.ctrl_btn_index.items():
            self.ctrl_btn_val[key] = self.joystick.get_button(value)
            logger.debug("{} = {}".format(key, self.ctrl_btn_val[key]))

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

    c = Controller(wait_interval_ms=25)

    c.run()
