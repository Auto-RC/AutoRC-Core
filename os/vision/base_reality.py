# ==================================================================================================
#                                        GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import threading
import numpy as np


# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
os_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

utility_dir = os_dir + r'/utility'

sys.path.append(utility_dir)

from logger import *


# ==================================================================================================
#                                         BASE REALITY
# ==================================================================================================

class base_reality:

    def load_imgs(self,path=None,rgb=None):

        if path != None:
            self.rgb_set = np.load(path)

    def load_img(self,img_num):

        if (img_num > self.rgb_set.shape[0]) or (img_num < 0):
            logger.info("Incorrect image num")

        self.rgb = self.rgb_set[img_num]


    # def clr_colors(self,r_start,r_end,g_start,g_end,b_start,b_end):


# ==================================================================================================
#                                          TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    br = base_reality()

    br.load_imgs(path=r"/Users/arnavgupta/car_data/iris-2019-02-09 17:48:41.158316.npy")

    br.load_img(0)

    logger.info("rgb shape: {}".format(br.rgb.shape))



