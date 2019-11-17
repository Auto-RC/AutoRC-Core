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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math

from autorc.vehicle.config import *

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

        self.road_history_len = 5
        self.road_history = []
        for i in range(0, self.road_history_len):
            self.road_history.append(None)

        self.prediction = None

    def init_filters(self):

        self.fil_rgb_l = np.array([146, 139, 0])
        self.fil_rgb_u = np.array([255, 255, 255])
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

    def need_correction(self):
        pass

    def p2l_dist(self, m, b, x, y):
        return np.abs(-m*x + y - b) / (((-m)**2 + 1)**.5)

    # def correct_splitter(self, splitter, splitter_c, thresh=20):
    #     print(self.prediction.splitter.midpoint, self.prediction.splitter.angle)
    #
    #     if not self.prediction.splitter.present:
    #         print("no correction, no splitter predicted")
    #         return splitter
    #
    #     if not splitter.present:
    #         if abs(self.prediction.splitter.midpoint) > 50:
    #             print("no correction, splitter entered")
    #             return splitter
    #
    #     if abs(splitter.midpoint - self.prediction.splitter.midpoint) < 10 and abs(splitter.angle - self.prediction.splitter.angle) < 25:
    #         print("no correction, splitter close to original")
    #         return splitter
    #
    #     print("correction needed")
    #
    #     m, b = self.prediction.splitter.cv_line
    #
    #     for i in range(len(splitter_c)):
    #         M = cv2.moments(splitter_c[i])
    #         x = int(M['m10'] / M['m00'])
    #         y = int(M['m01'] / M['m00'])
    #         if self.p2l_dist(m, b, x, y) > thresh:
    #             del splitter_c[i]
    #
    #     # for i in range(len(self.contours)):
    #     #     M = cv2.moments([self.contours[i]])
    #     #     x = int(M['m10'] / M['m00'])
    #     #     y = int(M['m01'] / M['m00'])
    #     #     if self.p2l_dist(m, b, x, y) < thresh:
    #     #         splitter_c.append(self.contours[i])
    #
    #     splitter = self.create_splitter(splitter_c, splitter)
    #
    #     return splitter

    def create_splitter(self, splitter_c, splitter):

        p1 = None
        p2 = None

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

        if p1 and p2:

            if (p2[0]-p1[0]) == 0:
                p1[0]+=1

            m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])

            if (m > 0.25) or (m < -0.25):

                print("m: {} p1: {} p2: {}".format(m, p1, p2))
                cv2.line(self.frame, p1, p2, (0, 0, 255), 1)

                cv_m = float(p2[1] - p1[1]) / float(p2[0] - p1[0] + 0.0001)
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

        return splitter

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

            if (p2[0] - p1[0]) == 0:
                p1[0] += 1

            m = float(p2[1] - p1[1]) / float(p2[0] - p1[0])

            if (m > 0.25) or (m < -0.25):

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


    def calculate_contour_slope(self, contour):

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        smallest_dist = 100
        n = 0

        for i in range(1, 4):
            dist = ((box[0][0] - box[i][0]) ** 2 + (box[0][1] - box[i][1]) ** 2) ** 0.5
            if dist < smallest_dist:
                smallest_dist = dist
                n = i

        p1 = ((box[0][0] + box[n][0]) / 2, (box[0][1] + box[n][1]) / 2)
        p2 = (sum([box[i][0] for i in range(1, 4) if i is not n]) / 2,
              sum([box[i][1] for i in range(1, 4) if i is not n]) / 2)
        dy = float(p2[1] - p1[1])
        dx = float(p2[0] - p1[0])
        p1 = (int(p1[0] + (dx * 200)), int(p1[1] + (dy * 200)))
        p2 = (int(p2[0] - (dx * 200)), int(p2[1] - (dy * 200)))

        # Calculating slope
        if (p2[0] - p1[0]) == 0:
            slope = (p2[1] - p1[1]) / 0.001
        else:
            slope = (p2[1] - p1[1])/(p2[0] - p1[0])

        # Calculating midpoint
        if slope == 0:
            mid_x = int((((self.frame.shape[0] / 2) - p1[1]) / 0.001) + p1[0]) - int(self.frame.shape[1] / 2)
        else:
            mid_x = int( (((self.frame.shape[0] / 2) - p1[1]) / slope) + p1[0]) - int(self.frame.shape[1] / 2)

        # Calculating the angle from slope
        raw_angle = np.rad2deg(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        # Shifting the angle quadrants appropriately to get the angle between -90 and 90
        if (raw_angle >= 0) and (raw_angle < 90):
            angle = 90 - raw_angle
        elif (raw_angle <= -90) and (raw_angle > -180):
            angle = 90-(raw_angle + 180)
        elif (raw_angle < 0) and (raw_angle > -90):
            angle = (raw_angle + 180)-90
        elif (raw_angle >= 90) and (raw_angle <= 180):
            angle = -(raw_angle - 90)

        angle = -1*angle # TODO: FIX THIS

        return slope, angle, p1, p2, mid_x

    def find_valid_contours(self, contour_limit):

        rows, cols = self.frame.shape[:2]

        # Initializing the contour filters
        contour_filters = []
        contour_filters.append({"min": contour_limit, "max": 200})
        # contour_filters.append({"min": 50, "max": 200})
        # contour_filters.append({"min": 80, "max": 200}) #95
        # contour_filters.append({"min": 110, "max": 130})
        # contour_filters.append({"min": 150, "max": 200})

        # Filtering the contours by size
        filtered_contours = []
        for c in self.contours:
            print(len(c))
            for filter in contour_filters:
                if (len(c) >= filter["min"]) and (len(c) < filter["max"]):
                    filtered_contours.append(c)

        # Sorting contours
        sorted_contours = []
        sorted_contours.append({"min":  -10, "max": -0.1, "contours": [], "data": []})
        sorted_contours.append({"min":  -25, "max":  -10, "contours": [], "data": []})
        sorted_contours.append({"min":  -45, "max":  -25, "contours": [], "data": []})
        sorted_contours.append({"min":  -65, "max":  -45, "contours": [], "data": []})
        sorted_contours.append({"min":  -89, "max":  -65, "contours": [], "data": []})
        sorted_contours.append({"min":    0, "max":   10, "contours": [], "data": []})
        sorted_contours.append({"min":   25, "max":   45, "contours": [], "data": []})
        sorted_contours.append({"min":   45, "max":   65, "contours": [], "data": []})
        sorted_contours.append({"min":   65, "max":   89, "contours": [], "data": []})

        # Used to filter on midpoints
        midpoints = []

        num_sorted_contour = 0
        for c in filtered_contours:

            # Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
            # In this case the 2D input point set is each contour
            rect = cv2.minAreaRect(c)

            # These are the 4 points defining the rotated rectangle provided to it
            box_points_float = cv2.boxPoints(rect)
            box_points_int = np.int0(box_points_float)
            slope, angle, p1, p2, mid_x = self.calculate_contour_slope(box_points_int)
            print("{} {}".format(slope, angle))

            # Filtering if mid points of lines are very close
            repeated_midpoint = False
            for midpoint in midpoints:
                if abs(mid_x-midpoint) < 15:
                    repeated_midpoint = True
            if repeated_midpoint == False:
                midpoints.append(mid_x)

            # Sorting the contours if midpoint is not repeated
            if not repeated_midpoint:
                for contour_category in sorted_contours:
                    if (angle >= contour_category["min"]) and (angle < contour_category["max"]):
                        if (len(contour_category["contours"]) == 0):
                            contour_category["contours"].append(rect)
                            contour_category["data"] = {"slope": slope, "angle": angle, "p1": p1, "p2": p2, "mid_x": mid_x}
                            num_sorted_contour += 1

            cv2.drawContours(image=self.frame, contours=[box_points_int], contourIdx=0, color=(0, 0, 255), thickness=1)

        for contour_category in sorted_contours:
            print("{} {}".format(contour_category, len(contour_category["contours"])))

        return self.frame, sorted_contours, num_sorted_contour

    def build_road(self, contours):

        # Extracting angels and midpoints
        angles = []
        midpoints = []
        p1_list = []
        p2_list = []
        for contour in contours:
            if len(contour["contours"]) > 0:
               angles.append(contour['data']['angle'])
               midpoints.append(contour['data']['mid_x'])
               p1_list.append(contour['data']['p1'])
               p2_list.append(contour['data']['p2'])

        if len(angles) > 0:

            # Determining turn type
            if sum(angles) > 10:
                turn = "right"
            elif sum(angles) < 10:
                turn = "left"
            else:
                turn = "straight"

            print(turn)

            if len(angles) == 2:

                if turn == "right":
                    if midpoints[0] > midpoints[1]:
                        splitter = TrackLine(True, angles[0], midpoints[0], [p1_list[0], p2_list[0]])
                        left_lane = TrackLine(True, angles[1], midpoints[1], [p1_list[1], p2_list[1]])
                    else:
                        splitter = TrackLine(True, angles[1], midpoints[1], [p1_list[1], p2_list[1]])
                        left_lane = TrackLine(True, angles[0], midpoints[0], [p1_list[0], p2_list[0]])
                    self.road = Road(None, splitter, left_lane, None)


                elif turn == "left":
                    if midpoints[0] < midpoints[1]:
                        splitter = TrackLine(True, angles[0], midpoints[0], [p1_list[0], p2_list[0]])
                        right_lane = TrackLine(True, angles[1], midpoints[1], [p1_list[1], p2_list[1]])
                    else:
                        splitter = TrackLine(True, angles[1], midpoints[1], [p1_list[1], p2_list[1]])
                        right_lane = TrackLine(True, angles[0], midpoints[0], [p1_list[0], p2_list[0]])
                    self.road = Road(None, splitter, None, right_lane)

            elif len(angles) == 3:

                print(midpoints)
                right_lane_index = midpoints.index(max(midpoints))
                left_lane_index = midpoints.index(min(midpoints))

                # There must be a better way of doing this?
                for i in range(0,3):
                    if (i != right_lane_index) and (i != left_lane_index):
                        splitter_index = i

                print(right_lane_index, left_lane_index, splitter_index)

                splitter = TrackLine(True, angles[splitter_index], midpoints[splitter_index], [p1_list[splitter_index], p2_list[splitter_index]])
                left_lane = TrackLine(True, angles[left_lane_index], midpoints[left_lane_index], [p1_list[left_lane_index], p2_list[left_lane_index]])
                right_lane = TrackLine(True, angles[right_lane_index], midpoints[right_lane_index], [p1_list[right_lane_index], p2_list[right_lane_index]])

                self.road = Road(None, splitter, left_lane, right_lane)

            elif len(angles) == 1:

                splitter = TrackLine(True, angles[0], midpoints[0], [p1_list[0], p2_list[0]])
                self.road = Road(None, splitter, None, None)

            else:

                self.road = Road(None, None, None, None)

        return self.road

    def process_road_history(self, road):

        for i in range(0, self.road_history_len-1):
            self.road_history[i+1] = self.road_history[i]

        self.road_history[0] = road

        if (self.road_history[1] != None) and (self.road_history[2] != None):

            if (self.road_history[0].splitter == None):

                # if (self.road_history[1].splitter != None) and (self.road_history[2] != None):
                #
                #     splitter_angle_shift = self.road_history[1].splitter.angle - self.road_history[2].splitter.angle
                #     splitter_midpoint_shift = self.road_history[1].splitter.midpoint - self.road_history[2].splitter.midpoint
                #     splitter_pred_angle = self.road_history[1].splitter.angle + splitter_angle_shift
                #     splitter_pred_midpoint = self.road_history[1].splitter.midpoint + splitter_midpoint_shift
                #     p1 = self.road_history[1].splitter.cv_line[0]
                #     p2 = self.road_history[1].splitter.cv_line[0]
                #     splitter_pred_p1 = (p1[0] + splitter_midpoint_shift, p1[1] + splitter_midpoint_shift)
                #     splitter_pred_p2 = (p2[0] + splitter_midpoint_shift, p2[1] + splitter_midpoint_shift)
                #
                #     splitter = TrackLine(True, splitter_pred_angle, splitter_pred_midpoint, [splitter_pred_p1, splitter_pred_p2])
                #
                #     self.road_history[0].splitter = splitter
                #     print("predicted next splitter")
                #
                # else:

                prev_found = False
                for i in range(1, self.road_history_len):
                    if self.road_history[i].splitter != None:
                        self.road_history[0].splitter = self.road_history[i].splitter
                        print('used previous splitter')
                        prev_found = True
                        break

            if (self.road_history[0].right_lane == None):

                # if (self.road_history[1].right_lane != None) and (self.road_history[2] != None):
                #
                #     right_lane_angle_shift = self.road_history[1].right_lane.angle - self.road_history[2].right_lane.angle
                #     right_lane_midpoint_shift = self.road_history[1].right_lane.midpoint - self.road_history[
                #         2].right_lane.midpoint
                #     right_lane_pred_angle = self.road_history[1].right_lane.angle + right_lane_angle_shift
                #     right_lane_pred_midpoint = self.road_history[1].right_lane.midpoint + right_lane_midpoint_shift
                #     p1 = self.road_history[1].right_lane.cv_line[0]
                #     p2 = self.road_history[1].right_lane.cv_line[0]
                #     right_lane_pred_p1 = (p1[0] + right_lane_midpoint_shift, p1[1] + right_lane_midpoint_shift)
                #     right_lane_pred_p2 = (p2[0] + right_lane_midpoint_shift, p2[1] + right_lane_midpoint_shift)
                #
                #     right_lane = TrackLine(True, right_lane_pred_angle, right_lane_pred_midpoint,
                #                          [right_lane_pred_p1, right_lane_pred_p2])
                #
                #     self.road_history[0].right_lane = right_lane
                #
                #     print("predicted next right lane")
                #
                # else:

                prev_found = False
                for i in range(1, self.road_history_len):
                    if self.road_history[i].right_lane != None:
                        self.road_history[0].right_lane = self.road_history[i].right_lane
                        print("used previous right lane")
                        prev_found = True
                        break

            if (self.road_history[0].left_lane == None):

                # if (self.road_history[1].left_lane != None) and (self.road_history[2] != None):
                #
                #     left_lane_angle_shift = self.road_history[1].left_lane.angle - self.road_history[2].left_lane.angle
                #     left_lane_midpoint_shift = self.road_history[1].left_lane.midpoint - self.road_history[2].left_lane.midpoint
                #     left_lane_pred_angle = self.road_history[1].left_lane.angle + left_lane_angle_shift
                #     left_lane_pred_midpoint = self.road_history[1].left_lane.midpoint + left_lane_midpoint_shift
                #     p1 = self.road_history[1].left_lane.cv_line[0]
                #     p2 = self.road_history[1].left_lane.cv_line[0]
                #     left_lane_pred_p1 = (p1[0] + left_lane_midpoint_shift, p1[1] + left_lane_midpoint_shift)
                #     left_lane_pred_p2 = (p2[0] + left_lane_midpoint_shift, p2[1] + left_lane_midpoint_shift)
                #
                #     left_lane = TrackLine(True, left_lane_pred_angle, left_lane_pred_midpoint, [left_lane_pred_p1, left_lane_pred_p2])
                #
                #     self.road_history[0].left_lane = left_lane
                #
                #     print("predicted next left lane")
                #
                # else:

                prev_found = False
                for i in range(1, self.road_history_len):
                    if self.road_history[i].left_lane != None:
                        self.road_history[0].left_lane = self.road_history[i].left_lane
                        print("used previous left lane")
                        prev_found = True
                        break

        return self.road_history[0]

    def draw_road(self, road):

        if road != None:

            if road.right_lane != None:
                cv2.line(self.frame, road.right_lane.cv_line[0], road.right_lane.cv_line[1], (255, 0, 0), 1)

            if road.left_lane != None:
                cv2.line(self.frame, road.left_lane.cv_line[0], road.left_lane.cv_line[1], (0, 255, 0), 1)

            if road.splitter != None:
                cv2.line(self.frame, road.splitter.cv_line[0], road.splitter.cv_line[1], (255, 255, 0), 1)

        return self.frame

    def process(self):

        self.frame = self.frame[40:73, :, :]

        # RGB TO HSV
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # HSV FILTER
        self.frame = self.filter_color(self.frame, self.fil_hsv_l, self.fil_hsv_u)

       # HSV TO RGB TO GRAY SCALE
        self.frame =  cv2.cvtColor(self.frame, cv2.COLOR_HSV2RGB)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        # Finding contours
        self.contours = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[contours_index]

        # GRAY TO RGB
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)


        contour_limit = 5
        retry_attempts = 5
        self.frame, contours, num_sorted_contours = self.find_valid_contours(contour_limit)
        while ((num_sorted_contours > 3) or (num_sorted_contours < 2)) and (retry_attempts < 5):

            if num_sorted_contours > 3:
                contour_limit += 10

            elif num_sorted_contours < 2:
                contour_limit -= 10

            self.frame, contours, num_sorted_contours = self.find_valid_contours(contour_limit)
            retry_attempts+=1

        self.road = self.build_road(contours)

        # self.road = self.process_road_history(current_road)

        self.frame = self.draw_road(self.road)

        #
        # splitter_c = []
        # lanes = []
        #
        # lanes = sorted(lanes, key=lambda x: cv2.contourArea(x), reverse=False)
        #
        # splitter = TrackLine(False, None, None, None)
        #
        # splitter = self.create_splitter(splitter_c, splitter)
        #
        # lanes = self.create_lanes(lanes)
        # left_lane, right_lane = self.assign_lanes(lanes, splitter)
        #
        # lines = [splitter, left_lane, right_lane]
        # self.road = Road(None, splitter, left_lane, right_lane)
        # self.road.vehicle = self.calc_vehicle(lines)

        return self.road

# ------------------------------------------------------------------------------
#                                      RETINA
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()
    # retina.set_calibration('splitter', [20,20,20], [255,255,250])
    # retina.set_calibration('lane', [40, 120, 21], [215, 155, 50])