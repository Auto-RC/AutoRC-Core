# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import sys
import numpy as np
import logging
# import opencv as cv2
from configparser import ConfigParser

# ------------------------------------------------------------------------------
#                                SETUP LOGGING
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
#                                   RETINA
# ------------------------------------------------------------------------------

class Retina():

    RHO = 1
    THETA = 90
    LINE_THRESHOLD = 20

    def __init__(self):

        self.frame = None
        self.enable_lines = False

        self.calibration_parser = ConfigParser()
        self.read_calibration()
        self.init_filters()

    def init_filters(self):

        self.fil_1_l = np.array([80, 0, 0])
        self.fil_1_u = np.array([110, 100, 255])

    def read_calibration(self):

        self.calibration_parser.read("calibration.ini")

        self.spl_rgb_lower_filter = [int(self.calibration_parser.get('splitter_parameters','l_r')),
                                     int(self.calibration_parser.get('splitter_parameters','l_g')),
                                     int(self.calibration_parser.get('splitter_parameters','l_b'))]

        self.spl_rgb_upper_filter = [int(self.calibration_parser.get('splitter_parameters', 'u_r')),
                                     int(self.calibration_parser.get('splitter_parameters', 'u_g')),
                                     int(self.calibration_parser.get('splitter_parameters', 'u_b'))]

        self.lane_rgb_lower_filter = [int(self.calibration_parser.get('splitter_parameters', 'l_r')),
                                      int(self.calibration_parser.get('splitter_parameters', 'l_g')),
                                      int(self.calibration_parser.get('splitter_parameters', 'l_b'))]

        self.lane_rgb_upper_filter = [int(self.calibration_parser.get('splitter_parameters', 'u_r')),
                                      int(self.calibration_parser.get('splitter_parameters', 'u_g')),
                                      int(self.calibration_parser.get('splitter_parameters', 'u_b'))]

    def set_calibration(self,type,lower_rgb,upper_rgb):

        self.calibration_parser.set('{}_parameters'.format(type),'l_r', str(lower_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type),'l_g', str(lower_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type),'l_b', str(lower_rgb[2]))

        self.calibration_parser.set('{}_parameters'.format(type), 'u_r',str(upper_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_g',str(upper_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_b',str(upper_rgb[2]))

        calibration_file = open("calibration.ini", "w")
        self.calibration_parser.write(calibration_file)

        logger.info("Set new calibration parameters for {} parameters".format(type))

    def hsv_transformation(self):

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        return self.frame

    def print_blue_hvs(self):

        blue = np.uint8([[[0, 0, 255]]])
        hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        print(hsv_blue)

    def filter_color(self, lower_rgb_range, upper_rgb_range):

        mask = cv2.inRange(self.frame, lower_rgb_range, upper_rgb_range)
        self.frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        return self.frame

    def detect_lanes(self):


        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges,self.RHO,np.pi/self.THETA,self.LINE_THRESHOLD)
        angles = []
        midpoints = []

        for line in lines[0:2]:
            for rho,theta in line:

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                if (x2-x1) > 0:
                    angles.append(np.arctan( (y2-y1)/(x2-x1) ))
                    midpoints.append([ (x2-x1)/2+x1 , (y2-y1)/2+y1 ])

                cv2.line(self.frame,(x1,y1),(x2,y2),(255,255,255),2)

        return { "frame" : self.frame , "lines" : lines , "angles" : angles , 'midpoints' : midpoints }

    def process(self):

        # This works for the initial images
        # fil_1_l = np.array([30, 0, 0])
        # fil_1_u = np.array([80, 105, 255])



        # self.filter_color(fil_1_l,fil_1_u)
        self.hsv_transformation()
        self.filter_color(self.fil_1_l,self.fil_1_u)

        if self.enable_lines:
            self.detect_lanes()

        return self.frame

# ------------------------------------------------------------------------------
#                                      RETINA
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()
    # retina.set_calibration('splitter', [20,20,20], [255,255,250])
    # retina.set_calibration('lane', [40, 120, 21], [215, 155, 50])

    retina.print_blue_hvs()