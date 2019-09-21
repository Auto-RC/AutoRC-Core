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

    def __init__(self, path, timestamp):

        self.vision_path = os.path.join(path, "oculus-{}.npy".format(timestamp))
        self.corti_path = os.path.join(path, "corti-{}.npy".format(timestamp))
        self.drive_path = os.path.join(path, "drive-{}.npy".format(timestamp))

        self.frames = []
        self.frame_index = 0

    def load(self):

        # Loading oculus data
        self.vision_frames = np.load(self.vision_path)
        print(self.vision_frames.shape)

        # Loading corti data
        self.corti_frames = np.load(self.corti_path)
        print(self.corti_frames.shape)

        # Loading drive data
        self.drive_frames = np.load(self.drive_path)
        print(self.drive_frames.shape)

        self.num_frames = self.vision_frames.shape[0]
        logger.debug("Number of frames in npy: {}".format(self.num_frames))

    def get_frame(self):

        # print("Image num: {}".format(self.img_num))
        frame = {"vision": self.vision_frames[self.frame_index],
                 "corti": self.corti_frames[self.frame_index],
                 "drive": self.drive_frames[self.frame_index],
                }

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