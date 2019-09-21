# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import os
from time import time
import numpy as np
import logging
from PIL import Image
import cv2

# ------------------------------------------------------------------------------
#                                SETUP LOGGING
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------
#                                    RECALL
# ------------------------------------------------------------------------------

class Recall():

    def __init__(self, path, timestamp, mode):

        if mode == "vision":
            self.path = os.path.join(path, "oculus-{}.npy".format(timestamp))

        if mode == "corti":
            self.path = os.path.join(path, "corti-{}.npy".format(timestamp))

        if mode == "drive":
            self.path = os.path.join(path, "drive-{}.npy".format(timestamp))

        self.frames = []
        self._frame_index = 0

    def load(self):

        # Loading oculus data
        self.frames = np.load(self.path)
        self.num_frames = self.frames.shape[0]
        logger.debug("Number of frames in npy: {}".format(self.num_frames))

    def get_frame_index(self):

        return self._frame_index

    def set_frame_index(self, index):

        self._frame_index = index

    def get_frame(self):

        frame = self.frames[self._frame_index]

        return frame

    def rgb_to_img(self, np_array) -> Image:

        """
        Convert an HxWx3 numpy array into an RGB Image
        """

        assert_msg = 'Input shall be a HxWx3 nparray'
        assert isinstance(np_array, np.ndarray), assert_msg
        assert len(np_array.shape) == 3, assert_msg
        assert np_array.shape[2] == 3, assert_msg

        img = Image.fromarray(np_array, 'RGB')
        return img

    def save_img(self, frame):

        img = self.rgb_to_img(frame)
        img.save(r'/media/sf_VM_Shared/autorc_data/test.png')

    def cv_read(self):

        return cv2.imread(r'/media/sf_VM_Shared/autorc_data/test.png')

# ------------------------------------------------------------------------------
#                                  UNIT TEST
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    recall = Recall(r"/media/sf_VM_Shared/autorc_data/oculus-2019-04-20 17;48;15.783634.npy")
    recall.load()