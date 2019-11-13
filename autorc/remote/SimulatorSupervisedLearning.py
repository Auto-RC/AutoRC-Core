"""
Simulator is able to load & replay collected racing data,
train and visualize models, callibrate vision algorithm configurations
"""

__author__ = "Anish Agarwal, Arnav Gupta"
__copyright__ = "Copyright 2019, AutoRC"
__version__ = "0.0.1"

import platform
import PIL
from PIL import ImageTk
import PIL.Image
import logging
from tkinter import *
from threading import Thread
import time
import platform
import numpy as np

from configparser import ConfigParser

from autorc.vehicle.vision.recall import Recall
from autorc.vehicle.controls.cerebellum_supervised_learning import CerebellumSupervisedLearning
from autorc.vehicle.cortex.cortex_advanced import CortexAdvanced

class Simulator(Thread):

    UI_HEIGHT = 775
    UI_WIDTH = 835

    IMG_WIDTH = 400
    IMG_HEIGHT = 200

    RESIZE_FACTOR = 3

    LOAD = True
    SAVE = False

    def __init__(self, data_path):

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.DEBUG)

        # Thread initialization
        Thread.__init__(self)

        # Flags
        self.video_active = False
        self.apply_retina = True

        # Data path for images
        self.data_path = data_path

        self.calibration_parser = ConfigParser()
        self.read_calibration()

        # Init recall
        self.init_recall()

        # Cerebellum Initialization
        cerebellum_update_interval_ms = 100
        cortex_update_interval_ms = 100

        drive = self.drive_recall
        corti = self.corti_recall
        oculus = self.vision_recall

        mode = True
        model_name = "M1_20191109"

        # Parameters
        self.loss_moving_avg = 0
        self.reward_moving_avg = 0

        self.cortex = CortexAdvanced(cortex_update_interval_ms, oculus, corti, drive)
        self.cortex.enable()
        self.cortex.start()

        time.sleep(1)
        self.cerebellum = CerebellumSupervisedLearning(cerebellum_update_interval_ms, drive, self.cortex, corti, model_name, mode, load=self.LOAD, save=self.SAVE)
        # self.cerebellum.auto = True
        # self.cerebellum.start()

        # Init UI
        self.init_ui()

        # Init vision controls
        self.init_img_controls()

        # Init Vision Features
        self.vectorized_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.vectorized_state_prev = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.init_vision_controls()
        self.init_vision_features()
        self.init_cerebellum_predictions()

        # Init RGB Controls
        self.init_vision_rgb_controls()

        # Init HVS Controls
        self.init_vision_hsv_controls()

        # Init Training Controls
        self.init_training_controls()

        # Init Canvas
        self.init_canvas()
        self.img_index = 0
        self.change_img(self.img_index)
        self.update_img()

    def init_recall(self):

        file_timestamp = "2019-11-09 20;54;16.865105"

        self.vision_recall = Recall(self.data_path, file_timestamp, "vision")
        self.vision_recall.load()

        self.corti_recall = Recall(self.data_path, file_timestamp, "corti")
        self.corti_recall.load()

        self.drive_recall = Recall(self.data_path, file_timestamp, "drive")
        self.drive_recall.load()

    def init_ui(self):

        self.ui = Tk()
        self.ui.resizable(width=False, height=False)
        self.ui.geometry("{}x{}".format(self.UI_WIDTH, self.UI_HEIGHT))

    def read_calibration(self):

        if 'Darwin' in platform.platform():
            self.calibration_parser.read(r"/Users/arnavgupta/AutoRC-Core/autorc/vehicle/vision/calibration.ini")
        else:
            self.calibration_parser.read(r"/home/veda/git/AutoRC-Core/autorc/vehicle/vision/calibration.ini")

        self.rgb_l = [
            int(self.calibration_parser.get('splitter_parameters', 'l_h')),
            int(self.calibration_parser.get('splitter_parameters', 'l_s')),
            int(self.calibration_parser.get('splitter_parameters', 'l_v'))]

        self.rgb_u = [
            int(self.calibration_parser.get('splitter_parameters', 'u_h')),
            int(self.calibration_parser.get('splitter_parameters', 'u_s')),
            int(self.calibration_parser.get('splitter_parameters', 'u_v'))]

        self.hsv_l = [
            int(self.calibration_parser.get('splitter_parameters', 'l_h')),
            int(self.calibration_parser.get('splitter_parameters', 'l_s')),
            int(self.calibration_parser.get('splitter_parameters', 'l_v'))]

        self.hsv_u = [
            int(self.calibration_parser.get('splitter_parameters', 'u_h')),
            int(self.calibration_parser.get('splitter_parameters', 'u_s')),
            int(self.calibration_parser.get('splitter_parameters', 'u_v'))]

    def init_vision_rgb_controls(self):

        rgb_filter_frame = LabelFrame(self.ui, text="RGB Filters", pady=15, padx=15)
        rgb_filter_frame.grid(row=0, column=10, rowspan=10, columnspan=15)

        r_upper_limit_label = Label(rgb_filter_frame, text="R Upper Lim", pady=5, padx=5)
        r_upper_limit_label.grid(row=1, column=0)
        self.r_upper_limit_var = DoubleVar()
        r_upper_limit = Scale(rgb_filter_frame, variable=self.r_upper_limit_var)
        r_upper_limit.grid(row=2, column=0)

        g_upper_limit_label = Label(rgb_filter_frame, text="G Upper Lim", pady=5, padx=5)
        g_upper_limit_label.grid(row=1, column=1)
        self.g_upper_limit_var = DoubleVar()
        g_upper_limit = Scale(rgb_filter_frame, variable=self.g_upper_limit_var)
        g_upper_limit.grid(row=2, column=1)

        b_upper_limit_label = Label(rgb_filter_frame, text="B Upper Lim", pady=5, padx=5)
        b_upper_limit_label.grid(row=1, column=2)
        self.b_upper_limit_var = DoubleVar()
        b_upper_limit = Scale(rgb_filter_frame, variable=self.b_upper_limit_var)
        b_upper_limit.grid(row=2, column=2)

        r_lower_limit_label = Label(rgb_filter_frame, text="R Lower Lim", pady=5, padx=5)
        r_lower_limit_label.grid(row=3, column=0)
        self.r_lower_limit_var = DoubleVar()
        r_lower_limit = Scale(rgb_filter_frame, variable=self.r_lower_limit_var)
        r_lower_limit.grid(row=4, column=0)

        g_lower_limit_label = Label(rgb_filter_frame, text="G Lower Lim", pady=5, padx=5)
        g_lower_limit_label.grid(row=3, column=1)
        self.g_lower_limit_var = DoubleVar()
        g_lower_limit = Scale(rgb_filter_frame, variable=self.g_lower_limit_var)
        g_lower_limit.grid(row=4, column=1)

        b_lower_limit_label = Label(rgb_filter_frame, text="B Lower Lim", pady=5, padx=5)
        b_lower_limit_label.grid(row=3, column=2)
        self.b_lower_limit_var = DoubleVar()
        b_lower_limit = Scale(rgb_filter_frame, variable=self.b_lower_limit_var)
        b_lower_limit.grid(row=4, column=2)

    def init_vision_hsv_controls(self):

        hsv_filter_frame = LabelFrame(self.ui, text="HSV Filters", pady=15, padx=15)
        hsv_filter_frame.grid(row=10, column=10, rowspan=10, columnspan=15)

        h_upper_limit_label = Label(hsv_filter_frame, text="H Upper Lim", pady=5, padx=5)
        h_upper_limit_label.grid(row=1, column=0)
        h_upper_limit_var = DoubleVar()
        self.h_upper_limit = Scale(hsv_filter_frame, variable=h_upper_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.h_upper_limit.set(self.hsv_u[0])
        self.h_upper_limit.grid(row=2, column=0)

        s_upper_limit_label = Label(hsv_filter_frame, text="S Upper Lim", pady=5, padx=5)
        s_upper_limit_label.grid(row=1, column=1)
        s_upper_limit_var = DoubleVar()
        self.s_upper_limit = Scale(hsv_filter_frame, variable=s_upper_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.s_upper_limit.set(self.hsv_u[1])
        self.s_upper_limit.grid(row=2, column=1)

        v_upper_limit_label = Label(hsv_filter_frame, text="V Upper Lim", pady=5, padx=5)
        v_upper_limit_label.grid(row=1, column=2)
        v_upper_limit_var = DoubleVar()
        self.v_upper_limit = Scale(hsv_filter_frame, variable=v_upper_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.v_upper_limit.set(self.hsv_u[2])
        self.v_upper_limit.grid(row=2, column=2)

        h_lower_limit_label = Label(hsv_filter_frame, text="H Lower Lim", pady=5, padx=5)
        h_lower_limit_label.grid(row=3, column=0)
        h_lower_limit_var = DoubleVar()
        self.h_lower_limit = Scale(hsv_filter_frame, variable=h_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.h_lower_limit.set(self.hsv_l[0])
        self.h_lower_limit.grid(row=4, column=0)

        s_lower_limit_label = Label(hsv_filter_frame, text="S Lower Lim", pady=5, padx=5)
        s_lower_limit_label.grid(row=3, column=1)
        s_lower_limit_var = DoubleVar()
        self.s_lower_limit = Scale(hsv_filter_frame, variable=s_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.s_lower_limit.set(self.hsv_l[1])
        self.s_lower_limit.grid(row=4, column=1)

        v_lower_limit_label = Label(hsv_filter_frame, text="V Lower Lim", pady=5, padx=5)
        v_lower_limit_label.grid(row=3, column=2)
        v_lower_limit_var = DoubleVar()
        self.v_lower_limit = Scale(hsv_filter_frame, variable=v_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.v_lower_limit.set(self.hsv_l[2])
        self.v_lower_limit.grid(row=4, column=2)

    def init_vision_controls(self):

        vision_controls_frame = LabelFrame(self.ui, pady=15, padx=15)
        vision_controls_frame.grid(row=10, column=0, rowspan=1, columnspan=10)

        self.toggle_vision = Button(vision_controls_frame, text="Toggle Vision", command=lambda: self.toggle_retina())
        self.toggle_vision.grid(row=0,column=0)

        self.toggle_lane_detection = Button(vision_controls_frame, text="Toggle Lane Detection", command=lambda: print("Enable Lane Detection"))
        self.toggle_lane_detection.grid(row=0, column=1)

        self.toggle_splitter_detection = Button(vision_controls_frame, text="Toggle Splitter Detection", command=lambda: print("Toggle Splitter Detection"))
        self.toggle_splitter_detection.grid(row=0, column=3)

    def init_vision_features(self):

        vision_feature_frame = Frame(self.ui)
        vision_feature_frame.grid(row=12, column=0, rowspan=8, columnspan=10)

        # Angles

        splitter_angle = Label(vision_feature_frame, text="Splitter Angle:", anchor="e", padx=15, pady=2)
        splitter_angle.grid(row=0, column=0)
        self.splitter_angle = StringVar()
        self.splitter_angle.set("Un-initialized")
        splitter_angle_var = Label(vision_feature_frame, textvariable=self.splitter_angle, anchor="w", padx=15, pady=2)
        splitter_angle_var.grid(row=0, column=1)

        left_lane_angle = Label(vision_feature_frame, text="Left Lane Angle:", anchor="w")
        left_lane_angle.grid(row=1, column=0)
        self.left_lane_angle = StringVar()
        self.left_lane_angle.set("Un-initialized")
        left_lane_angle_var = Label(vision_feature_frame, textvariable=self.left_lane_angle, anchor="w", padx=15, pady=2)
        left_lane_angle_var.grid(row=1, column=1)

        right_lane_angle = Label(vision_feature_frame, text="Right Lane Angle:", anchor="w")
        right_lane_angle.grid(row=2, column=0)
        self.right_lane_angle = StringVar()
        self.right_lane_angle.set("Un-initialized")
        right_lane_angle_var = Label(vision_feature_frame, textvariable=self.right_lane_angle, anchor="w", padx=15, pady=2)
        right_lane_angle_var.grid(row=2, column=1)

        vehicle_angle = Label(vision_feature_frame, text="Vehicle Angle:", anchor="w")
        vehicle_angle.grid(row=3, column=0)
        self.vehicle_angle = StringVar()
        self.vehicle_angle.set("Un-initialized")
        vehicle_angle_var = Label(vision_feature_frame, textvariable=self.vehicle_angle, anchor="w", padx=15, pady=2)
        vehicle_angle_var.grid(row=3, column=1)

        # Feature Existence

        splitter_present = Label(vision_feature_frame, text="Splitter Present:", anchor="e")
        splitter_present.grid(row=0, column=3)
        self.splitter_present = StringVar()
        self.splitter_present.set("Un-initialized")
        splitter_present_var = Label(vision_feature_frame, textvariable=self.splitter_present, anchor="w", padx=15, pady=2)
        splitter_present_var.grid(row=0, column=4)

        left_lane_present = Label(vision_feature_frame, text="Left Lane Present:", anchor="w")
        left_lane_present.grid(row=1, column=3)
        self.left_lane_present = StringVar()
        self.left_lane_present.set("Un-initialized")
        left_lane_present_var = Label(vision_feature_frame, textvariable=self.left_lane_present, anchor="w", padx=15, pady=2)
        left_lane_present_var.grid(row=1, column=4)

        right_lane_present = Label(vision_feature_frame, text="Right Lane Present:", anchor="w")
        right_lane_present.grid(row=2, column=3)
        self.right_lane_present = StringVar()
        self.right_lane_present.set("Un-initialized")
        right_lane_present_var = Label(vision_feature_frame, textvariable=self.right_lane_present, anchor="w", padx=15, pady=2)
        right_lane_present_var.grid(row=2, column=4)

        vehicle_offroad = Label(vision_feature_frame, text="Vehicle Offroad:", anchor="w")
        vehicle_offroad.grid(row=3, column=3)
        self.vehicle_offroad = StringVar()
        self.vehicle_offroad.set("Un-initialized")
        vehicle_offroad_var = Label(vision_feature_frame, textvariable=self.vehicle_offroad, anchor="w", padx=15, pady=2)
        vehicle_offroad_var.grid(row=3, column=4)

        # Positions

        left_lane_position = Label(vision_feature_frame, text="Left Lane Position:", anchor="w")
        left_lane_position.grid(row=4, column=3)
        self.left_lane_position = StringVar()
        self.left_lane_position.set("Un-initialized")
        left_lane_position_var = Label(vision_feature_frame, textvariable=self.left_lane_position, anchor="w", padx=15,pady=2)
        left_lane_position_var.grid(row=4, column=4)

        right_lane_position = Label(vision_feature_frame, text="Right Lane Position:", anchor="w")
        right_lane_position.grid(row=5, column=3)
        self.right_lane_position = StringVar()
        self.right_lane_position.set("Un-initialized")
        right_lane_position_var = Label(vision_feature_frame, textvariable=self.right_lane_position, anchor="w", padx=15, pady=2)
        right_lane_position_var.grid(row=5, column=4)

        splitter_position = Label(vision_feature_frame, text="Splitter Position:", anchor="w")
        splitter_position.grid(row=4, column=0)
        self.splitter_position = StringVar()
        self.splitter_position.set("Un-initialized")
        splitter_position_var = Label(vision_feature_frame, textvariable=self.splitter_position, anchor="w", padx=15, pady=2)
        splitter_position_var.grid(row=4, column=1)

        vehicle_position = Label(vision_feature_frame, text="Vehicle Position:", anchor="w")
        vehicle_position.grid(row=5, column=0)
        self.vehicle_position = StringVar()
        self.vehicle_position.set("Un-initialized")
        vehicle_position_var = Label(vision_feature_frame, textvariable=self.vehicle_position, anchor="w", padx=15, pady=2)
        vehicle_position_var.grid(row=5, column=1)

    def init_cerebellum_predictions(self):

        cerebellum_predictions_frame = Frame(self.ui)
        cerebellum_predictions_frame.grid(row=20, column=0, rowspan=8, columnspan=10)

        computed_throttle = Label(cerebellum_predictions_frame, text="Computed Throttle:", anchor="e", padx=15, pady=2)
        computed_throttle.grid(row=0, column=0)
        self.computed_throttle = StringVar()
        self.computed_throttle.set("Un-initialized")
        computed_throttle_var = Label(cerebellum_predictions_frame, textvariable=self.computed_throttle, anchor="w", padx=15, pady=2)
        computed_throttle_var.grid(row=0, column=1)

        computed_steering = Label(cerebellum_predictions_frame, text="Computed Steering:", anchor="w")
        computed_steering.grid(row=0, column=2)
        self.computed_steering = StringVar()
        self.computed_steering.set("Un-initialized")
        computed_steering_var = Label(cerebellum_predictions_frame, textvariable=self.computed_steering, anchor="w", padx=15, pady=2)
        computed_steering_var.grid(row=0, column=3)

        user_throttle = Label(cerebellum_predictions_frame, text="User Throttle:", anchor="e", padx=15, pady=2)
        user_throttle.grid(row=1, column=0)
        self.user_throttle = StringVar()
        self.user_throttle.set("Un-initialized")
        user_throttle_var = Label(cerebellum_predictions_frame, textvariable=self.user_throttle, anchor="w", padx=15, pady=2)
        user_throttle_var.grid(row=1, column=1)

        user_steering = Label(cerebellum_predictions_frame, text="User Steering:", anchor="w")
        user_steering.grid(row=1, column=2)
        self.user_steering = StringVar()
        self.user_steering.set("Un-initialized")
        user_steering_var = Label(cerebellum_predictions_frame, textvariable=self.user_steering, anchor="w", padx=15, pady=2)
        user_steering_var.grid(row=1, column=3)

        loss = Label(cerebellum_predictions_frame, text="Loss:", anchor="w")
        loss.grid(row=2, column=0)
        self.loss = StringVar()
        self.loss.set("Un-initialized")
        loss_var = Label(cerebellum_predictions_frame, textvariable=self.loss, anchor="w", padx=15, pady=2)
        loss_var.grid(row=2, column=1)

        reward = Label(cerebellum_predictions_frame, text="Reward:", anchor="w")
        reward.grid(row=2, column=2)
        self.reward = StringVar()
        self.reward.set("Un-initialized")
        reward_var = Label(cerebellum_predictions_frame, textvariable=self.reward, anchor="w", padx=15, pady=2)
        reward_var.grid(row=2, column=3)

        loss_moving_average = Label(cerebellum_predictions_frame, text="Avg Loss:", anchor="w")
        loss_moving_average.grid(row=3, column=0)
        self.loss_moving_average = StringVar()
        self.loss_moving_average.set("Un-initialized")
        loss_moving_average_var = Label(cerebellum_predictions_frame, textvariable=self.loss_moving_average, anchor="w", padx=15, pady=2)
        loss_moving_average_var.grid(row=3, column=1)

        reward_moving_average = Label(cerebellum_predictions_frame, text="Avg Reward:", anchor="w")
        reward_moving_average.grid(row=3, column=2)
        self.reward_moving_average = StringVar()
        self.reward_moving_average.set("Un-initialized")
        reward_moving_average_var = Label(cerebellum_predictions_frame, textvariable=self.reward_moving_average, anchor="w", padx=15, pady=2)
        reward_moving_average_var.grid(row=3, column=3)

        batches_trained = Label(cerebellum_predictions_frame, text="Batches Trained:", anchor="w")
        batches_trained.grid(row=4, column=0)
        self.batches_trained = StringVar()
        self.batches_trained.set("Un-initialized")
        batches_trained_var = Label(cerebellum_predictions_frame, textvariable=self.batches_trained, anchor="w", padx=15, pady=2)
        batches_trained_var.grid(row=4, column=1)

        exploration_rate = Label(cerebellum_predictions_frame, text="Exploration Rate:", anchor="w")
        exploration_rate.grid(row=4, column=2)
        self.exploration_rate = StringVar()
        self.exploration_rate.set("Un-initialized")
        exploration_rate_var = Label(cerebellum_predictions_frame, textvariable=self.exploration_rate, anchor="w", padx=15, pady=2)
        exploration_rate_var.grid(row=4, column=3)

    def init_img_controls(self):

        img_controls_frame = LabelFrame(self.ui, pady=15, padx=15)
        img_controls_frame.grid(row=9, column=0, rowspan=1, columnspan=10)

        self.next = Button(img_controls_frame, text="Next", command=lambda: self.next_img())
        self.next.grid(row=0, column=0)

        self.previous = Button(img_controls_frame, text="Previous", command=lambda: self.previous_img())
        self.previous.grid(row=0, column=1)

        self.play = Button(img_controls_frame, text="Play", command=lambda: self.start_video())
        self.play.grid(row=0, column=2)

        self.stop = Button(img_controls_frame, text="Stop", command=lambda: self.stop_video())
        self.stop.grid(row=0, column=3)

    def init_training_controls(self):

        training_controls_frame = LabelFrame(self.ui, pady=15, padx=15)
        training_controls_frame.grid(row=11, column=0, rowspan=1, columnspan=10)

        self.train = Button(training_controls_frame, text="Train", command=lambda: self.start_training())
        self.train.grid(row=0, column=0)

        self.stop_train = Button(training_controls_frame, text="Stop Training", command=lambda: self.stop_training_thread())
        self.stop_train.grid(row=0, column=1)

    def init_canvas(self):

        self.canvas = Canvas(self.ui, width=self.IMG_WIDTH, height=self.IMG_HEIGHT)
        self.canvas.grid(row=0,column=0, rowspan=10, columnspan=10)
        self.canvas_img = self.canvas.create_image((10, 10), image=(), anchor='nw')

    def change_img(self, img_index):

        self.get_frame(img_index)

        if self.apply_retina == False:

            self.raw = self.raw[40:80:, :]
            self.img = ImageTk.PhotoImage(self.resize_im(self.raw))

            self.update_img()
            self.logger.info("Image {} opened".format(self.img_index))

        elif self.apply_retina == True:

            self.processed = self.cortex.retina.frame
            self.img = ImageTk.PhotoImage(self.resize_im(self.processed))

            self.update_vision()
            self.update_predictions()

            self.update_img()
            self.logger.info("Image {} opened".format(self.img_index))

    def get_frame(self, frame_num):

        # Getting the vision frame
        self.vision_recall.set_frame_index(frame_num)
        self.raw = self.vision_recall.get_frame()
        self.img = ImageTk.PhotoImage(self.resize_im(self.raw))

        # Getting the corti frame
        self.corti_recall.set_frame_index(frame_num)
        self.corti_frame = self.corti_recall.get_frame()

        # Getting the drive frame
        self.drive_recall.set_frame_index(frame_num)
        self.drive_frame = self.drive_recall.get_frame()

    def update_img(self):

        self.canvas.itemconfigure(self.canvas_img, image=self.img)

    def resize_im(self, im):

        im = self.vision_recall.rgb_to_img(im)
        return im.resize((128 * self.RESIZE_FACTOR, 40 * self.RESIZE_FACTOR), PIL.Image.NEAREST)
        # return im

    def next_img(self):

        if self.img_index == self.vision_recall.num_frames - 1:
            self.img_index = 0
        else:
            self.img_index += 1

        self.change_img(self.img_index)

    def previous_img(self):

        if self.img_index == 0:
            self.logger.info("First image reached")
        else:
            self.img_index -= 1
            self.change_img(self.img_index)


    def start_video(self):

        self.video_active = True
        self.video = Thread(target=self.video_thread, args=())
        self.video.start()

    def stop_video(self):

        self.video_active = False

    def video_thread(self):

        while self.video_active:

            self.img_index += 1
            if self.img_index >= self.vision_recall.num_frames:
                self.img_index = 0

            self.change_img(self.img_index)

            time.sleep(0.05)

    def start_training(self):

        self.training_active = True
        self.training = Thread(target=self.train_thread, args=())
        self.training.start()

    def stop_training_thread(self):

        self.training_active = False

    def train_thread(self):

        while self.training_active:

            self.img_index += 1
            if self.img_index >= self.vision_recall.num_frames:
                self.img_index = 0

            self.get_frame(self.img_index)

            self.processed = self.cortex.retina.frame

            self.update_vision()
            self.update_predictions()

    def update_hsv(self):

        self.cortex.retina.fil_hsv_u[0] = int(self.h_upper_limit.get())
        self.cortex.retina.fil_hsv_u[1] = int(self.s_upper_limit.get())
        self.cortex.retina.fil_hsv_u[2] = int(self.v_upper_limit.get())

        self.cortex.retina.fil_hsv_l[0] = int(self.h_lower_limit.get())
        self.cortex.retina.fil_hsv_l[1] = int(self.s_lower_limit.get())
        self.cortex.retina.fil_hsv_l[2] = int(self.v_lower_limit.get())

        # self.cortex.retina.set_calibration(self.TYPE, self.cortex.retina.fil_rgb_l, self.cortex.retina.fil_rgb_u)

        self.change_img(self.img_index)

    def toggle_retina(self):

        if not self.apply_retina:
            self.apply_retina = True

        elif self.apply_retina:
            self.apply_retina = False

        self.change_img(self.img_index)

    def update_vision(self):

        try:
            self.left_lane_angle.set('%.2f' % self.cortex.observation_space['left_lane_angle'])
        except:
            self.left_lane_angle.set(self.cortex.observation_space['left_lane_angle'])

        try:
            self.right_lane_angle.set('%.2f' % self.cortex.observation_space['right_lane_angle'])
        except:
            self.right_lane_angle.set(self.cortex.observation_space['right_lane_angle'])

        try:
            self.splitter_angle.set('%.2f' % self.cortex.observation_space['splitter_angle'])
        except:
            self.splitter_angle.set(self.cortex.observation_space['splitter_angle'])

        try:
            self.vehicle_angle.set('%.2f' % self.cortex.observation_space['vehicle_angle'])
        except:
            self.vehicle_angle.set(self.cortex.observation_space['vehicle_angle'])

        self.left_lane_present.set(self.cortex.observation_space['left_lane_present'])
        self.right_lane_present.set(self.cortex.observation_space['right_lane_present'])
        self.splitter_present.set(self.cortex.observation_space['splitter_present'])
        self.vehicle_offroad.set(self.cortex.observation_space['vehicle_offroad'])

        try:
            self.left_lane_position.set('%.2f' % self.cortex.observation_space['left_lane_position'])
        except:
            self.left_lane_position.set(self.cortex.observation_space['left_lane_position'])

        try:
            self.right_lane_position.set('%.2f' % self.cortex.observation_space['right_lane_position'])
        except:
            self.right_lane_position.set(self.cortex.observation_space['right_lane_position'])

        try:
            self.splitter_position.set('%.2f' % self.cortex.observation_space['splitter_position'])
        except:
            self.splitter_position.set(self.cortex.observation_space['splitter_position'])

        try:
            self.vehicle_position.set('%.2f' % self.cortex.observation_space['vehicle_position'])
        except:
            self.vehicle_position.set(self.cortex.observation_space['vehicle_position'])

    def update_predictions(self):

        # Updating the state
        self.raw_state = self.cortex.get_raw_state() / 255.0
        self.cerebellum.update_state(self.raw_state)

        # Getting the machine computed action
        action = self.cerebellum.compute_controls()[0]
        print('Action', action)
        self.computed_throttle.set('%.2f' % action[1])
        self.computed_steering.set('%.2f' % action[0])

        print('Drive frame:', self.drive_frame)
        # Getting the user action
        self.user_throttle.set('%.2f' % self.drive_frame[1])
        self.user_steering.set('%.2f' % self.drive_frame[0])

        # Computing reward and loss
        # reward = self.cortex.compute_reward(action["action"][0], action["action"][1])
        reward = self.cortex.compute_reward(action[0], action[1])
        self.cerebellum.remember(self.raw_state, self.drive_frame, self.cortex.observation_space['terminal'])
        avg_loss = self.cerebellum.experience_replay()
        self.loss.set('%.8f' %  avg_loss)

        self.loss_moving_avg = 0.8*self.loss_moving_avg + 0.2*avg_loss
        self.loss_moving_average.set('%.8f' % self.loss_moving_avg)

        batches_trained = self.cerebellum.get_batches_trained()
        self.batches_trained.set('%.2f' % batches_trained)

    def run(self):

        self.ui.mainloop()

if __name__ == '__main__':


    print(platform.platform())
    print("Platform: {}".format(platform.platform()))

    if 'Darwin' in platform.platform():
        data_path = "/Users/arnavgupta/car_data/raw_npy/"
    else:
        data_path = r"/home/veda/git/AutoRC-Core/autorc/sample_data"

    simulator = Simulator(data_path)
    simulator.run()