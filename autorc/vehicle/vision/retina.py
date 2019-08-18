# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import sys
import numpy as np
import logging
import cv2
from configparser import ConfigParser
import itertools
import time

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
    LINE_THRESHOLD = 15

    def __init__(self):

        self.frame = None
        self.frame_l = None
        self.frame_c = None
        self.frame_r = None
        self.frames = []
        self.enable_lines = True
        self.mode = 'RGB'

        self.angles = [0,0,0]
        self.midpoints = [(0,0),(0,0),(0,0)]

        self.calibration_parser = ConfigParser()
        # self.read_calibration()
        self.init_filters()

    def init_filters(self):

        self.fil_rgb_l = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.fil_rgb_u = np.array([[255, 255, 255],[255, 255, 255],[255, 255, 255]])
        self.fil_hsv_l = np.array([0, 0, 0])
        self.fil_hsv_u = np.array([255, 255, 255])

    def read_calibration(self):

        self.calibration_parser.read("/Users/arnavgupta/auto-rc_poc/os/vision/calibration.ini")

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

        self.spl_hsv_lower_filter = [int(self.calibration_parser.get('splitter_parameters', 'l_h')),
                                     int(self.calibration_parser.get('splitter_parameters', 'l_s')),
                                     int(self.calibration_parser.get('splitter_parameters', 'l_v'))]

        self.spl_hsv_upper_filter = [int(self.calibration_parser.get('splitter_parameters', 'u_h')),
                                     int(self.calibration_parser.get('splitter_parameters', 'u_s')),
                                     int(self.calibration_parser.get('splitter_parameters', 'u_v'))]

        self.lane_hsv_lower_filter = [
            int(self.calibration_parser.get('splitter_parameters', 'l_h')),
            int(self.calibration_parser.get('splitter_parameters', 'l_s')),
            int(self.calibration_parser.get('splitter_parameters', 'l_v'))]

        self.lane_hsv_upper_filter = [
            int(self.calibration_parser.get('splitter_parameters', 'u_h')),
            int(self.calibration_parser.get('splitter_parameters', 'u_s')),
            int(self.calibration_parser.get('splitter_parameters', 'u_v'))]

    def set_calibration(self,type,lower_rgb,upper_rgb):

        self.calibration_parser.set('{}_parameters'.format(type),'l_r', str(lower_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type),'l_g', str(lower_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type),'l_b', str(lower_rgb[2]))

        self.calibration_parser.set('{}_parameters'.format(type), 'u_r',str(upper_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_g',str(upper_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_b',str(upper_rgb[2]))

        self.calibration_parser.set('{}_parameters'.format(type), 'l_h', str(lower_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type), 'l_s', str(lower_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type), 'l_v', str(lower_rgb[2]))

        self.calibration_parser.set('{}_parameters'.format(type), 'u_h', str(upper_rgb[0]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_s', str(upper_rgb[1]))
        self.calibration_parser.set('{}_parameters'.format(type), 'u_v', str(upper_rgb[2]))

        calibration_file = open("calibration.ini", "w")
        self.calibration_parser.write(calibration_file)

        logger.info("Set new calibration parameters for {} parameters".format(type))

    def hsv_transformation(self):

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        return self.frame

    def print_blue_hvs(self):

        blue = np.uint8([[[0, 0, 255]]])
        hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        # print(hsv_blue)

    def filter_color(self, im, lower_rgb_range, upper_rgb_range):

        mask = cv2.inRange(im, lower_rgb_range, upper_rgb_range)
        im = cv2.bitwise_and(im, im, mask=mask)
        return im

        # print(self.fil_rgb_l, self.fil_rgb_u)

    def detect_lanes(self):

        for i in range(len(self.frames)):
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges,self.RHO,np.pi/self.THETA,self.LINE_THRESHOLD)
            angles = []
            midpoints = []
            if lines is not None:
                for line in lines[0:1]:
                    for rho,theta in line:

                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        theta *= 180/np.pi
                        if 90 <= theta < 180:
                            theta -= 180

                        self.angles[i] = theta
                        self.midpoints[i] = rho

                        # if (x2-x1) > 0:
                        #     self.angles[i] = np.arctan( (y2-y1)/(x2-x1) ) * 180/np.pi
                        #     self.midpoints[i] = ((x2-x1)/2+x1 , (y2-y1)/2+y1 )

                        cv2.line(self.frames[i],(x1,y1),(x2,y2),(255,255,255),2)
            else:
                self.angles[i] = None
                self.midpoints[i] = None

        # return { "frame" : self.frame , "lines" : lines , "angles" : angles , 'midpoints' : midpoints }


    def process(self):

        # This works for the initial images
        # fil_1_l = np.array([30, 0, 0])
        # fil_1_u = np.array([80, 105, 255])

        # print(self.frame.shape)
        self.frame = self.frame[40:73, :, :]

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.frame = self.filter_color(self.frame, self.fil_hsv_l, self.fil_hsv_u)

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_HSV2RGB)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)


        print(self.frame[0])

        _, contours, hierarchy = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # self.hsv_s_u_filter()



        # self.filter_color(fil_1_l,fil_1_u)
        # print(self.enable_lines, self.mode)
        # self.rgb_red_filter()
        # self.hsv_s_u_filter()
        # for i in range(len(self.frames)):
        #     self.frames[i] = self.filter_color(self.frames[i],
        #                                        self.fil_rgb_l[i],
        #                                        self.fil_rgb_u[i])


        # for i in range(len(self.frames)):
        #     # self.frames[i] = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2HSV)
        #
        #     self.frames[i] = self.filter_color(self.frames[i],
        #                                        self.fil_hsv_l,
        #                                        self.fil_hsv_u)

        if self.enable_lines:
            self.detect_lanes()


        # if self.mode == 'HSV':
        #     return self.frame
        # elif self.mode == 'RGB':
        #     return rgb_frame

        # print(np.mean([self.angles[0],self.angles[2]]))
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)
        # print(self.frame.shape)

        splitter = []
        lanes = []


        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            edges = 0
            leftmost = tuple(c[c[:, :, 0].argmin()][0])
            rightmost = tuple(c[c[:, :, 0].argmax()][0])
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])

            if leftmost[0] < 3:
                edges += 1
            if rightmost[0] > self.frame.shape[1] - 5:
                edges += 1
            if topmost[1] < 5:
                edges += 1


            if rect[1][0] != 0 and rect[1][1] != 0:
                extent = cv2.contourArea(c)/(rect[1][0] * rect[1][1])
                # print(extent,)
            else:
                extent = 0



            if edges == 2:
                # print("edged", rect[0], rect[1])
                # cv2.drawContours(self.frame, [box], 0, (255, 0, 0), 1)
                lanes.append(c)
            elif extent < 0.5:
                # print("not rect", rect[0], rect[1], topmost, rightmost, bottommost, leftmost)
                pass

            elif rect[1][0] * 7 > rect[1][1] and rect[1][1] * 7 > rect[1][0] and rect[1][0] > 1 and rect[1][1] * 8 > 1:
                cv2.drawContours(self.frame, [box], 0, (0, 0, 255), 1)
                # print("allowed", rect[0], rect[1])
                splitter.append(c)
            else:
                # print("removed", rect[0], rect[1], topmost, rightmost, bottommost, leftmost)
                cv2.drawContours(self.frame, [box], 0, (0, 255, 0), 1)

                # cons.append([rect[0], cv2.contourArea(c)])

        lanes = sorted(lanes, key=lambda x: cv2.contourArea(x), reverse=False)

        rows, cols = self.frame.shape[:2]
        self.lane_eqs = []
        self.splitter_eq = None
        for c in lanes:
            if cv2.contourArea(c) > 3:
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((self.frame.shape[1] - x) * vy / vx) + y)
                p1 = (self.frame.shape[1]-1,righty)
                p2 = (0,lefty)
                cv2.line(self.frame, p1, p2, (255, 0, 0), 1)
                p1 = (p1[0], (self.frame.shape[0] - 1)-p1[1])
                p2 = (p2[0], (self.frame.shape[0] - 1)-p2[1])
                m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
                x_inter = (p1[1] / m) + p1[0]
                angle = np.tan(1/m)
                print(m, x_inter, angle)
                self.lane_eqs.append([angle, x_inter, m])



        if len(splitter) > 1:
            c = np.array(list(itertools.chain.from_iterable(splitter)))
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((self.frame.shape[1] - x) * vy / vx) + y)
            p1 = (self.frame.shape[1] - 1, righty)
            p2 = (0, lefty)
            cv2.line(self.frame, p1, p2, (0, 0, 255), 1)
            p1 = (p1[0], (self.frame.shape[0] - 1) - p1[1])
            p2 = (p2[0], (self.frame.shape[0] - 1) - p2[1])
            m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
            x_inter = (p1[1] / m) + p1[0]
            angle = np.tan(1 / m)
            print(m, x_inter, angle)
            self.splitter_eq = [angle, x_inter, m]
        elif len(splitter) > 0:
            rect = cv2.minAreaRect(splitter[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            smallest_dist = 100
            n = 0
            for i in range(1, 4):
                dist = ((box[0][0] - box[i][0]) ** 2 + (box[0][1] - box[i][1]) ** 2) ** 0.5
                if ((box[0][0] - box[i][0]) ** 2 + (box[0][1] - box[i][1]) ** 2) ** 0.5 < smallest_dist:
                    smallest_dist = 0

            print(box)

        return 0, 0


# ------------------------------------------------------------------------------
#                                      RETINA
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()
    # retina.set_calibration('splitter', [20,20,20], [255,255,250])
    # retina.set_calibration('lane', [40, 120, 21], [215, 155, 50])

    retina.print_blue_hvs()