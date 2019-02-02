# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import numpy as np
import datetime

# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

from logger import *

# ==================================================================================================
#                                           IRIS
# ==================================================================================================

class Memory:

    def init_package(self, modules=[]):

        # Initializing a key in data package for every module in modules
        # ------------------------------------------------------------------------------------------
        self.modules = modules
        self.data_package = dict()
        for module in self.modules:
            self.data_package[module] = []

        self.timestamp = datetime.datetime.now()
        logger.debug('Created a data package at {}'.format(self.timestamp))

    def add(self, data_packet):

        for module in self.modules:
            self.data_package[module].append(data_packet[module])

    def save(self):

        for module in self.modules:
            np.save("data/{}-{}".format(module,self.timestamp), np.array(self.data_package[module]))
