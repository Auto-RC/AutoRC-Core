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
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
#                                     RGB
# ------------------------------------------------------------------------------

class Retina():

    RHO = 1
    THETA = 90
    LINE_THRESHOLD = 20

    def __init__(self):

        self.data_path = r'C:\Users\Veda Sadhak\Desktop\auto-rc_data'

    # ----------------------------------------------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------------------------------------------

    def set_frame(self,frame):

        self.origin = frame
        self.frame = frame
        self.x = frame.shape[1]
        self.y = frame.shape[2]

    def load_npy(self,file_name):

        # Where the raw image data is stored
        self.npy_path = os.path.join(self.data_path, 'raw_npy', file_name)

        # Where the raw images will be stored
        self.raw_img_dir = os.path.join(self.data_path, 'raw_png')
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
            logger.info("Made dir {}".format(self.raw_img_dir))

        # Where the raw images will be stored
        self.proc_img_dir = os.path.join(self.data_path, 'processed_png')
        if not os.path.exists(self.proc_img_dir):
            os.makedirs(self.proc_img_dir)
            logger.info("Made dir {}".format(self.proc_img_dir))

        # Loading the npy data package
        self.npy = np.load(self.npy_path)
        self.num_images = self.npy.shape[0]
        logger.info("Number of images in npy: {}".format(self.num_images))

    # ----------------------------------------------------------------------------------------------
    # SAVING
    # ----------------------------------------------------------------------------------------------

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

    def save_cv_img(self, cv_img, dir):

        cv2.imwrite(dir, cv_img)

    # ----------------------------------------------------------------------------------------------
    # LINE DETECTION
    # ----------------------------------------------------------------------------------------------

    def line_detection_unit_test(self):

        # Converting the images
        starting_time = time()


        self.line_detection_dir = os.path.join(self.data_path,r'line_detector')
        if not os.path.exists(self.line_detection_dir):
            os.makedirs(self.line_detection_dir)
            logger.info("Made dir {}".format(self.line_detection_dir))

        for index, original_img in enumerate(self.npy):

            logger.debug("Converting image {}...".format(index))

            self.set_img(original_img)

            self.filter_colors(np.array([0,45,60]), np.array([100, 250, 250]))
            self.find_lines()
            self.save_cv_img(self.img,os.path.join(self.line_detection_dir, "img_{}.png".format(index)))

            logger.debug("Done converting image {}.".format(index))

        ending_time = time()
        time_taken = ending_time - starting_time
        logger.info("Time taken to process and convert {} images: {}".format(self.num_images,time_taken))

    def detect_lines(self):

        lines = cv2.HoughLines(self.frame,self.RHO,np.pi/self.THETA,self.LINE_THRESHOLD)
        angles = []
        midpoints = []

        for line in lines[0]:
            for rho,theta in line:

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                angles.append(np.arctan( (y2-y1)/(x2-x1) ))
                midpoints.append([ (x2-x1)/2+x1 , (y2-y1)/2+y1 ])

                self.frame = cv2.line(self.frame,(x1,y1),(x2,y2),(255,255,255),2)

        return { "lines" : lines , "angles" : angles , 'midpoints' : midpoints}

    # ----------------------------------------------------------------------------------------------
    # COLOR DETECTION
    # ----------------------------------------------------------------------------------------------

    def color_detection_unit_test(self):

        # Converting the images
        starting_time = time()

        r_l = 0
        g_l = 45
        b_l = 60
        # b_u = 255

        for b_u in range(100,255,50):
            for r_u in range(50, 255, 50):
                for g_u in range(50, 255, 50):

                    self.filter_detection_dir = os.path.join(self.data_path,r'filter_detector','{}-{}-{}_{}-{}-{}'.format(r_l,g_l,b_l,r_u,g_u,b_u))
                    if not os.path.exists(self.filter_detection_dir):
                        os.makedirs(self.filter_detection_dir)
                        logger.info("Made dir {}".format(self.filter_detection_dir))

                    for index, original_img in enumerate(self.npy):

                        logger.debug("Converting image {}...".format(index))

                        self.set_img(original_img)

                        # converted_img = self.rgb_to_img(self.origin)
                        # converted_img.save(os.path.join(self.raw_img_dir, "img_{}.png".format(index)), 'PNG')

                        self.filter_colors(np.array([r_l,g_l,b_l]),np.array([r_u,g_u,b_u]))
                        self.save_cv_img(self.img,os.path.join(self.filter_detection_dir, "img_{}.png".format(index)))

                        logger.debug("Done converting image {}.".format(index))

                    ending_time = time()
                    time_taken = ending_time - starting_time
                    logger.info("Time taken to process and convert {} images: {}".format(self.num_images,time_taken))

    def filter_colors(self,lower_rgb_range,upper_rgb_range):

        self.frame = cv2.inRange(self.frame,lower_rgb_range,upper_rgb_range)

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()

    retina.load_npy(file_name='oculus-2019-04-19 18;16;06.500887.npy')
    retina.line_detection_unit_test()





