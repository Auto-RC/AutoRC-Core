"""
Class LapHistory
"""

import logging

class LapHistory():

    """
    Stores the history of the track
    for error correction in retina
    """

    def __init__(self, memory_size):

        """
        :param memory_size: number of track snapshots to be stores
        """

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        self.memory_size = memory_size

        self.lap = []

    def add_road_snapshot(self, road):

        """
        :param road: Road object containing the left lane, splitter, right lane, and vehicle
        at a certain time
        :return: None, adds road object to self.lap
        """

        if len(self.lap) == self.memory_size:
            del self.lap[0]

        self.lap.append(road)

    def predict(self):
        """
        :return: most recent state of the road
        Will later predict where the road should be
        """

        return self.lap[-1]