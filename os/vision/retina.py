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

class Retina():

    def __init__(self):

        self.data_path = r'C:\Users\Veda Sadhak\Desktop\auto-rc_data'

    def set_img(self,img):

        self.origin = img
        self.img = img
        self.x = img.shape[1]
        self.y = img.shape[2]

    def filter_colors(self,lower_rgb_range,upper_rgb_range):

        """
        Only keeps the colors between the specified color range, that is,
        between lower rgb and upper rgb
        """

        for x in range(0,self.x):
            for y in range(0,self.y):

                orignal_rgb = self.img[x][y]

                keep_red = (orignal_rgb[0] > lower_rgb_range[0]) and \
                           (orignal_rgb[0] < upper_rgb_range[0])

                keep_green = (orignal_rgb[1] > lower_rgb_range[1]) and \
                             (orignal_rgb[1] < upper_rgb_range[1])

                keep_blue = (orignal_rgb[2] > lower_rgb_range[2]) and \
                            (orignal_rgb[2] < upper_rgb_range[2])

                if (keep_red) and (keep_green) and (keep_blue):
                    new_rgb = orignal_rgb
                else:
                    new_rgb = [0,0,0]

                self.img[x][y] = new_rgb

    def load_npy(self,file_name):

        # Where the raw image data is stored
        self.npy_path = os.path.join(self.data_path, 'raw_npy', file_name)

        # Where the raw images will be stored
        self.img_dir = os.path.join(self.data_path, 'processed_png')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            logger.info("Made dir {}".format(self.img_dir))

        # Loading the npy data package
        self.npy = np.load(self.npy_path)
        self.num_images = self.npy.shape[0]
        logger.info("Number of images in npy: {}".format(self.num_images))

    def run_npy(self):

        # Converting the images
        starting_time = time()
        for index, original_img in enumerate(self.npy):

            logger.debug("Converting image {}...".format(index))

            self.img = original_img
            self.x = self.img.shape[0]
            self.y = self.img.shape[1]
            self.filter_colors([0,0,20],[255,255,200])

            converted_img = self.rgb_to_img(self.img)
            converted_img.save(os.path.join(self.img_dir, "img_{}.png".format(index)), 'PNG')

            logger.debug("Done converting image {}.".format(index))

        ending_time = time()
        time_taken = ending_time - starting_time
        logger.info("Time taken to process and convert {} images: {}".format(self.num_images,time_taken))

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

    retina = Retina()

    retina.load_npy(file_name='iris.npy')
    retina.run_npy()
    # retina.conv_npy_package(file_name='iris_new.npy', type='processed')





