# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import os
from time import time
import numpy as np
import logging
from PIL import Image

# ------------------------------------------------------------------------------
#                                SETUP LOGGING
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
#                                     RGB
# ------------------------------------------------------------------------------

class RGB():

    def __init__(self):

        self.data_path = r'C:\Users\Veda Sadhak\Desktop\auto-rc_data'

    def conv_npy_package(self,file_name, type='raw'):

        # Where the raw image date is stored
        npy_path = os.path.join(self.data_path,type+'_npy',file_name)

        # Where the raw images will be stored
        img_dir = os.path.join(self.data_path,type+'_png',file_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            logger.info("Made dir {}".format(img_dir))

        # Loading the npy data package
        img_package = np.load(npy_path)
        num_images = img_package.shape[0]
        logger.info("Number of images in file: {}".format(num_images))

        # Converting the images
        starting_time = time()
        for index , single_img in enumerate(img_package):
            converted_img = self.rgb_to_img(single_img)
            converted_img.save(os.path.join(img_dir, "img_{}.png".format(index)), 'PNG')
        ending_time = time()
        time_taken = ending_time - starting_time
        logger.info("Time taken to convert {} images: {}".format(num_images,time_taken))

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

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    rgb = RGB()

    rgb.conv_npy_package(file_name='iris.npy',type='raw')
    rgb.conv_npy_package(file_name='iris_new.npy', type='processed')





