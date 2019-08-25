# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------

import logging
from configparser import ConfigParser
import itertools
import platform
import cv2
import numpy as np
from autorc.vehicle.cortex.environment.environment import *
from autorc.vehicle.cortex.environment.lap_history import LapHistory

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

        self.enable_lines = True
        self.mode = 'RGB'

        self.calibration_parser = ConfigParser()
        # self.read_calibration()
        self.init_filters()

        self.lane_width = 50
        self.split_m = 0

        self.road = None

        self.prediction = None

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

    def filter_color(self, im, lower_rgb_range, upper_rgb_range):

        mask = cv2.inRange(im, lower_rgb_range, upper_rgb_range)
        im = cv2.bitwise_and(im, im, mask=mask)
        return im

        # print(self.fil_rgb_l, self.fil_rgb_u)

    def calc_vehicle(self, lines):
        angle = 0
        position = 0
        offroad = False
        lanes = 0
        for line in lines:
            if line.present:
                angle += line.angle + (line.midpoint * 0.8)
                lanes += 1
        angle /= -lanes

        if self.road.splitter.present:
            right_l = 0
            left_l = 0
            if self.road.right_lane.present:
                right_l = abs(self.road.right_lane.midpoint - self.road.splitter.midpoint)
            if self.road.left_lane.present:
                left_l = abs(self.road.splitter.midpoint - self.road.left_lane.midpoint)
            if right_l != 0 or left_l != 0:
                self.lane_width = max(right_l, left_l)
            position = -self.road.splitter.midpoint / self.lane_width

        elif self.road.right_lane.present:
            position = (self.lane_width - self.road.right_lane.midpoint) / self.lane_width

        elif self.road.left_lane.present or position < 1:
            position = -(self.road.left_lane.midpoint + self.lane_width) / self.lane_width

        else:
            offroad = True

        if abs(position) > 1.2:
            offroad = True

        vehicle = Vehicle(angle, position, None, None, offroad)
        return vehicle

    def update_line(self, obj, angle, midpoint, cv_line):
        obj.present = True
        obj.angle = angle
        obj.midpoint = midpoint
        obj.cv_line = cv_line
        return obj

    def need_correction(self):
        pass

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


        if 'Darwin' in platform.platform():
            contours = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        else:
            contours = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


        # if self.mode == 'HSV':
        #     return self.frame
        # elif self.mode == 'RGB':
        #     return rgb_frame

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)

        splitter_c = []
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
            if topmost[1] < 3:
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

            else:
                cv2.drawContours(self.frame, [box], 0, (0, 0, 255), 1)
                # print("allowed", rect[0], rect[1])
                splitter_c.append(c)

        lanes = sorted(lanes, key=lambda x: cv2.contourArea(x), reverse=False)

        rows, cols = self.frame.shape[:2]
        splitter = TrackLine(False, None, None, None)
        right_lane = TrackLine(False, None, None, None)
        left_lane = TrackLine(False, None, None, None)

        if len(splitter_c) > 1:
            c = np.array(list(itertools.chain.from_iterable(splitter_c)))
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((self.frame.shape[1] - x) * vy / vx) + y)
            p1 = (self.frame.shape[1] - 1, righty)
            p2 = (0, lefty)
        elif len(splitter_c) > 0:
            rect = cv2.minAreaRect(splitter_c[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            smallest_dist = 100
            n = 0
            for i in range(1, 4):
                dist = ((box[0][0] - box[i][0]) ** 2 + (box[0][1] - box[i][1]) ** 2) ** 0.5
                if dist < smallest_dist:
                    smallest_dist = dist
                    n = i
            p1 = ((box[0][0] + box[n][0])/2, (box[0][1] + box[n][1])/2)
            p2 = (sum([box[i][0] for i in range(1, 4) if i is not n]) / 2, sum([box[i][1] for i in range(1, 4) if i is not n]) / 2)
            dy = float(p2[1] - p1[1])
            dx = float(p2[0] - p1[0])
            p1 = (int(p1[0] + (dx * 200)), int(p1[1] + (dy * 200)))
            p2 = (int(p2[0] - (dx * 200)), int(p2[1] - (dy * 200)))

        if splitter_c:
            cv2.line(self.frame, p1, p2, (0, 0, 255), 1)
            cv_m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
            cv_b = p1[1] - (cv_m * p1[0])
            p1 = (p1[0], (self.frame.shape[0] - 1) - p1[1])
            p2 = (p2[0], (self.frame.shape[0] - 1) - p2[1])
            try: m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
            except: m = 0
            m += 0.001
            x_inter = int((((self.frame.shape[0] / 2) - p1[1]) / m) + p1[0]) - int(self.frame.shape[1] / 2)
            angle = (np.arctan(1 / m) * 180 / np.pi)
            # print("splitter", x_inter, angle)
            splitter = self.update_line(splitter, angle, x_inter, [cv_m, cv_b])
            self.split_m = splitter.midpoint

        for c in lanes:
            if cv2.contourArea(c) > 3:
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((self.frame.shape[1] - x) * vy / vx) + y)
                p1 = (self.frame.shape[1]-1,righty)
                p2 = (0,lefty)
                cv_m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
                cv_b = p1[1] - (cv_m * p1[0])
                cv2.line(self.frame, p1, p2, (255, 0, 0), 1)
                p1 = (p1[0], (self.frame.shape[0] - 1)-p1[1])
                p2 = (p2[0], (self.frame.shape[0] - 1)-p2[1])
                m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])
                x_inter = int((((self.frame.shape[0]/2)-p1[1]) / m) + p1[0]) - int(self.frame.shape[1] / 2)
                angle = (np.arctan(1 / m) * 180 / np.pi)
                if x_inter < self.split_m:
                    left_lane = self.update_line(left_lane, angle, x_inter, [cv_m, cv_b])
                    # print("left", x_inter, angle)
                else:
                    right_lane = self.update_line(right_lane, angle, x_inter, [cv_m, cv_b])
                    # print("right", x_inter, angle)

        lines = [splitter, left_lane, right_lane]
        self.road = Road(None, splitter, left_lane, right_lane)
        self.road.vehicle = self.calc_vehicle(lines)

        # print("veh", self.road.vehicle.angle, self.road.vehicle.position, self.lane_width)

        return self.road

# ------------------------------------------------------------------------------
#                                      RETINA
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()
    # retina.set_calibration('splitter', [20,20,20], [255,255,250])
    # retina.set_calibration('lane', [40, 120, 21], [215, 155, 50])

    retina.print_blue_hvs()