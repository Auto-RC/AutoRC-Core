# ==================================================================================================
#                                           GLOBAL IMPORTS
# ==================================================================================================

import os
from shutil import copyfile
import logging

# ==================================================================================================
#                                            LOGGER SETUP
# ==================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ==================================================================================================
#                                              AUTO COPY
# ==================================================================================================

class Auto_Copy():

    def __init__(self, source_dir, destination_dir):

        self.src_dir = source_dir
        self.dst_dir = destination_dir

        logger.info("Source dir: {}".format(source_dir))
        logger.info("Destination dir: {}".format(source_dir))

    def list_dst(self):

        self.dst_file_paths = os.listdir(self.dst_dir)

    def list_src(self):

        self.src_file_paths = os.listdir(self.src_dir)

    def run(self):

        self.list_dst()
        self.list_src()

        for file_path in self.src_file_paths:
            if file_path not in self.dst_file_paths:

                logger.info("Copying file {}...".format(file_path))
                copyfile(os.path.join(self.src_dir,file_path),self.dst_dir)

        logger.info("Done copying files")

# ==================================================================================================
#                                              UNIT TEST
# ==================================================================================================

if __name__ == '__main__':

    source_dir = r'/home/arnavgupta/data'
    destination_dir = r'/mnt/pidrive/data'

    auto_copy = Auto_Copy(source_dir,destination_dir)

    auto_copy.run()