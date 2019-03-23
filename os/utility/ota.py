# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import os
import logging
import git


# ------------------------------------------------------------------------------
#                                SETUP LOGGING
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
#                             Over-the-Air Updates
# ------------------------------------------------------------------------------

class OTA():

    def __init__(self):

        self.current_dir = os.path.dirname(os.path.realpath(__file__))

    def update(self):

        g = git.cmd.Git(self.current_dir)
        g.pull()


# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    ota = OTA()
    ota.update()

