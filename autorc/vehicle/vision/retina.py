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
from itertools import chain, combinations

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

        # self.calibration_parser = ConfigParser()
        # self.read_calibration()
        self.init_filters()

        self.lane_width = 50
        self.split_m = 0

        self.road = None

        self.prediction = Road(None, None, None, None)

    def init_filters(self):

        self.fil_rgb_l = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.fil_rgb_u = np.array([[255, 255, 255],[255, 255, 255],[255, 255, 255]])
        self.fil_hsv_l = np.array([0, 0, 0])
        self.fil_hsv_u = np.array([255, 255, 255])

    def read_calibration(self):

        self.calibration_parser.read("/Users/arnavgupta/AutoRC-Core/autorc/vehicle/vision/calibration.ini")

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
            if line:
                if line.present:
                    angle += line.angle + (line.midpoint * 0.8)
                    lanes += 1
        if lanes == 0:
            vehicle = Vehicle(None, None, None, None, True)
            return vehicle
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

    def p2l_dist(self, m, b, x, y):
        return np.abs(-m*x + y - b) / (((-m)**2 + 1)**.5)


    def chain_conts(self, ordered_c, centers, initial, direction=0, line=None):

        angle, x_inter, cv_line = self.fitCont([ordered_c[initial]], line)
        new_line = TrackLine(angle and x_inter and cv_line, angle, x_inter, cv_line)
        # self.drawLine(cv_line, (127, 0, 127))

        if direction == 0:
            inds = [initial, self.chain_conts(ordered_c, centers, initial, -1, new_line)[0], self.chain_conts(ordered_c, centers, initial, 1, new_line)[0]]
            print(inds)
            return inds

        elif direction == 1:
            for i in range(initial+1, len(ordered_c)):
                dist = self.p2l_dist(cv_line[0], cv_line[1], centers[i][0], centers[i][1])
                if dist < abs(centers[i][1] - centers[initial][1]) * 2:
                    return [i, self.chain_conts(ordered_c, centers, i, 1, new_line)[0]]
            return [initial]

        elif direction == -1:
            for i in range(initial - 1, -1, -1):
                dist = self.p2l_dist(cv_line[0], cv_line[1], centers[i][0], centers[i][1])
                if dist < abs(centers[i][1] - centers[initial][1]) / 2:
                    return [i, self.chain_conts(ordered_c, centers, i, -1, new_line)[0]]
            return [initial]



    def fix_splitter(self, splitter, splitter_c, left_lane, right_lane):

        if len(splitter_c) == 0:
            return splitter

        # centers = []
        for i, c in enumerate(splitter_c):
            M = cv2.moments(c)

            if M['m00'] == 0:
                M['m00'] += 0.000001
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # centers.append([cx, cy])

            if left_lane.present:
                cv_m, cv_b = left_lane.cv_line
                if cx < (cy - cv_b) / cv_m:
                    del splitter_c[i]
                    # centers[i] = None
            if right_lane.present:
                cv_m, cv_b = right_lane.cv_line
                if cx > (cy - cv_b) / cv_m:
                    del splitter_c[i]
                    # centers[i] = None

        # splitter_c = [i for i in splitter_c if type(i) == np.ndarray]
        # centers = [i for i in centers if not (i == None)]

        # try:
        #
        #     ordered_c = [x for _, x in sorted(zip([y[1] for y in centers], splitter_c), reverse=True)]
        #     centers = [x for _, x in sorted(zip([y[1] for y in centers], centers), reverse=True)]
        #
        #     initial = 0
        #
        #     inds = self.chain_conts(ordered_c, centers, initial, direction=0)
        #     inds = [i for i in inds if i is not -1]
        #
        #     splitter_c = [ordered_c[i] for i in inds]
        #
        # except:
        #     pass

        angle, x_inter, cv_line = self.fitCont(splitter_c)
        if not (angle or x_inter or cv_line):
            return splitter

        self.drawLine(cv_line, (0, 0, 255))

        splitter = self.update_line(splitter, angle, x_inter, cv_line)
        self.split_m = splitter.midpoint

        return splitter

    def create_splitter(self, splitter_c, splitter):

        angle, x_inter, cv_line = self.fitCont(splitter_c)

        if not (angle or x_inter or cv_line):
            return splitter

        splitter = self.update_line(splitter, angle, x_inter, cv_line)
        self.split_m = splitter.midpoint

        return splitter

    def fitLine(self, p1, p2):

        dy = float(p2[1] - p1[1])
        dx = float(p2[0] - p1[0])
        p1 = (int(p1[0] + (dx * 200)), int(p1[1] + (dy * 200)))
        p2 = (int(p2[0] - (dx * 200)), int(p2[1] - (dy * 200)))

        cv_m, cv_b = self.linCoeffs(p1, p2)

        p1 = (p1[0], (self.frame.shape[0] - 1) - p1[1])
        p2 = (p2[0], (self.frame.shape[0] - 1) - p2[1])

        m, b = self.linCoeffs(p1, p2)

        if m == 0:
            angle = 90

            if self.prediction.splitter:
                if self.prediction.splitter.angle and self.prediction.splitter.angle > 0:
                    x_inter = int((((self.frame.shape[0] / 2) - p1[1]) / 0.00001) + p1[0]) - int(self.frame.shape[1] / 2)
                else:
                    x_inter = int((((self.frame.shape[0] / 2) - p1[1]) / -0.00001) + p1[0]) - int(self.frame.shape[1] / 2)
                    angle = -90
            else:
                x_inter = int((((self.frame.shape[0] / 2) - p1[1]) / 0.00001) + p1[0]) - int(self.frame.shape[1] / 2)

        else:
            x_inter = int((((self.frame.shape[0] / 2) - p1[1]) / (m + 0.00001)) + p1[0]) - int(self.frame.shape[1] / 2)
            angle = (np.arctan(1 / m) * 180 / np.pi)

        return angle, x_inter, [cv_m, cv_b]

    def fitCont(self, c, line=None):

        if line == None:
            line = self.prediction.splitter

        if not c or c == []:
            return None, None, None

        elif len(c) == 1:
            rect = cv2.minAreaRect(c[0])
            box = np.int0(cv2.boxPoints(rect))
            box = np.append(box, box[0]).reshape((5, 2))
            centers = [((box[i][0] + box[i+1][0])/2, (box[i][1] + box[i+1][1])/2) for i in range(0, 4)]
            l1 = self.fitLine(centers[0], centers[2])
            l2 = self.fitLine(centers[1], centers[3])
            if self.evaluate(l1, line) > self.evaluate(l2, line):
                return l1
            else:
                return l2

        else:
            c = np.array(list(itertools.chain.from_iterable(c)))
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((self.frame.shape[1] - x) * vy / vx) + y)
            p1 = (self.frame.shape[1] - 1, righty)
            p2 = (0, lefty)
            return self.fitLine(p1, p2)

    def evaluate(self, fitline, baseline):
        if baseline:
            if baseline.present:
                strength = 1 - (0.75 * abs(fitline[0] - baseline.angle) / 180 + 0.25 * abs(fitline[1] - baseline.midpoint) / self.frame.shape[1])
                return strength

        return 1 - abs(fitline[0]) / 360

    def linCoeffs(self, p1, p2):
        if p2[0] == p1[0]:
            m = 1000000
        else:
            m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])

        b = p1[1] - (m * p1[0])

        return m, b

    def drawLine(self, cv_line, color):
        p1 = (0, int(cv_line[1]))
        p2 = (1000, int(cv_line[0]*1000 + cv_line[1]))
        cv2.line(self.frame, p1, p2, color)

    def create_lanes(self, lanes):
        lanes = [c for c in lanes if cv2.contourArea(c) > 3]

        for i, c in enumerate(lanes):

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
            m += 0.001
            try:
                x_inter = int((((self.frame.shape[0]/2)-p1[1]) / m) + p1[0]) - int(self.frame.shape[1] / 2)
            except:
                x_inter = 1000
            angle = (np.arctan(1 / m) * 180 / np.pi)
            lanes[i] = TrackLine(True, angle, x_inter, [cv_m, cv_b])

        return lanes

    def assign_lanes(self, lanes, splitter):

        right_lane = TrackLine(False, None, None, None)
        left_lane = TrackLine(False, None, None, None)


        for lane in lanes:
            if lane.midpoint < self.split_m:
                if not left_lane.present:
                    left_lane = lane
                else:
                    if lane.midpoint > left_lane.midpoint:
                        left_lane = lane
            if lane.midpoint > self.split_m:
                if not right_lane.present:
                    right_lane = lane
                else:
                    if lane.midpoint < right_lane.midpoint:
                        right_lane = lane

        if splitter.present is False and left_lane.present and right_lane.present:
            if abs(left_lane.midpoint-self.split_m) > abs(right_lane.midpoint-self.split_m):
                right_lane = TrackLine(False, None, None, None)
            else:
                left_lane = TrackLine(False, None, None, None)


        return [left_lane, right_lane]

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

        kernel = np.ones((2, 5), np.uint8)
        self.frame = cv2.dilate(self.frame, kernel, iterations=1)

        if 'Darwin' in platform.platform():
            self.contours = cv2.findContours(self.frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        else:
            self.contours = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]


        # if self.mode == 'HSV':
        #     return self.frame
        # elif self.mode == 'RGB':
        #     return rgb_frame

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)

        rows, cols = self.frame.shape[:2]

        splitter_c = []
        lanes = []
        lanes_misc = [[],[]]

        for i, c in enumerate(self.contours):


            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            lane = False
            leftmost = tuple(c[c[:, :, 0].argmin()][0])
            rightmost = tuple(c[c[:, :, 0].argmax()][0])
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])

            # print(i, leftmost, rightmost, topmost, bottommost)
            # cv2.drawContours(self.frame, c, -1, (0, 255, 255), 2)

            if topmost[1] < 1:
                if bottommost[1] > rows - 5:
                    lane = True
                elif rightmost[0] > cols - 2:
                    if leftmost[1] < rightmost[1]:
                        lane = True
                elif leftmost[0] < 2:
                    if rightmost[1] < leftmost[1]:
                        lane = True
            elif bottommost[1] > rows - 5:
                if rightmost[1] > 1:
                    if leftmost[0] < 1 and topmost[0] < 1:
                        lane = True
                elif leftmost[1] > 1:
                    if rightmost[0] > cols - 2 and topmost[0] > cols - 2:
                        lane = True
            elif leftmost[0] > 1 and rightmost[0] > self.frame.shape[1] - 1:
                lane = True

            if rect[1][0] != 0 and rect[1][1] != 0:
                extent = cv2.contourArea(c)/(rect[1][0] * rect[1][1])
                # print(extent,)
            else:
                extent = 0

            if lane:
                # print("edged", rect[0], rect[1])
                # cv2.drawContours(self.frame, [box], 0, (255, 0, 0), 1)
                lanes.append(c)

            elif extent < 0.5:
                M = cv2.moments(c)

                if M['m00'] == 0:
                    M['m00'] += 0.000001
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # cv2.circle(self.frame, (cx, cy), 0, (255, 0, 255), 5)
                #
                # if self.prediction.left_lane.present:
                #     m, b = self.prediction.left_lane.cv_line
                #     self.drawLine([m,b], (255, 0, 255))
                #     print(i, self.p2l_dist(m, b, cx, cy))
                #     if self.p2l_dist(m, b, cx, cy) < 10:
                #         lanes_misc[0].append(c)
                #         cv2.circle(self.frame, (cx, cy), 0, (255, 0, 255), 5)
                #         cv2.drawContours(self.frame, c, -1, (255, 255, 0))
                #
                # elif self.prediction.right_lane.present:
                #     m, b = self.prediction.right_lane.cv_line
                #     if self.p2l_dist(m, b, cx, cy) < 5:
                #         lanes_misc[1].append(c)


            else:
                cv2.drawContours(self.frame, [box], 0, (0, 0, 255), 1)
                # print("allowed", rect[0], rect[1])
                splitter_c.append(c)

        lanes = sorted(lanes, key=lambda x: cv2.contourArea(x), reverse=False)

        splitter = TrackLine(False, None, None, None)

        splitter = self.create_splitter(splitter_c, splitter)

        # if self.prediction:
        #     splitter = self.correct_splitter(splitter, splitter_c)
        # else:
        #     print("no prediction")

        lanes = self.create_lanes(lanes)
        left_lane, right_lane = self.assign_lanes(lanes, splitter)

        splitter = self.fix_splitter(splitter, splitter_c, left_lane, right_lane)

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
