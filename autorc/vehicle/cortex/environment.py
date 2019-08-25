"""
Holds objects used to define objects in the environment
"""
class TrackLine:

    """
    Class for left lane, right lane, and splitter
    """

    def __init__(self, present, angle, midpoint, cv_line):
        """
        :param present: if the line is visble or not (boolean)
        :param angle: angle of the line (-1 to 1)
        :param midpoint: midpoint of the line
        :param cv_line: line for calculating error in retina
        """
        self.present = present
        self.angle = angle
        self.midpoint = midpoint
        self.cv_line = cv_line

class Vehicle:

    """
    Class defining vehicles position relative to the track
    """

    def __init__(self, angle, position, x_acc, z_acc, offroad):
        """
        :param angle: vehicle's angle
        :param position: vehicle's position relative to the splitter (around -1 to 1)
        :param x_acc: from imu
        :param z_acc: from imu
        :param offroad: calculated based on state machine (boolean)
        """
        self.angle = angle
        self.position = position
        self.x_acc = x_acc
        self.z_acc = z_acc
        self.offroad = offroad

class Road:

    """
    Class packaging all the elements of the track together
    """

    def __init__(self, vehicle, splitter, left_lane, right_lane):
        """
        :param vehicle:
        :param splitter:
        :param left_lane:
        :param right_lane:
        """
        self.vehicle = vehicle
        self.splitter = splitter
        self.left_lane = left_lane
        self.right_lane = right_lane
