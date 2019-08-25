"""
Simulator is able to load & replay collected racing data,
train and visualize models, callibrate vision algorithm configurations
"""

__author__ = "Anish Agarwal, Arnav Gupta"
__copyright__ = "Copyright 2019, AutoRC"
__version__ = "0.0.1"


import os
import PIL
from PIL import ImageTk
import PIL.Image
import logging
from tkinter import *
from threading import Thread
import time
import platform

from autorc.vehicle.vision.recall import Recall
from autorc.vehicle.vision.retina import Retina
from autorc.vehicle.controls.cerebellum_advanced import CerebellumAdvanced
from autorc.vehicle.cortex.cortex_advanced import CortexAdvanced

class Simulator(Thread):

    UI_HEIGHT = 625
    UI_WIDTH = 785

    IMG_WIDTH = 400
    IMG_HEIGHT = 200

    RESIZE_FACTOR = 3

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

        # Retina initializaion
        self.retina = Retina()

        # Init recall
        self.init_recall()

        # Cerebellum Initialization
        cerebellum_update_interval_ms = 10
        cortex_update_interval_ms = 100
        controller = None
        corti = None
        mode = True
        oculus = self.recall
        model_name = "Test"
        self.cortex = CortexAdvanced(cortex_update_interval_ms, oculus, corti, controller)
        self.cortex.enable()
        self.cortex.start()
        time.sleep(1)
        self.cerebellum = CerebellumAdvanced(cerebellum_update_interval_ms, controller, self.cortex, corti, model_name, mode, load=False, train=False)
        self.cerebellum.auto = True
        self.cerebellum.start()

        # Init UI
        self.init_ui()

        # Init vision controls
        self.init_img_controls()

        # Init Vision Features
        self.init_vision_controls()
        self.init_vision_features()

        # Init RGB Controls
        self.init_vision_rgb_controls()

        # Init HVS Controls
        self.init_vision_hsv_controls()

        # Init Canvas
        self.init_canvas()
        self.img_index = 24
        self.change_img(self.img_index)
        self.update_img()

    def init_recall(self):

        self.recall = Recall(self.data_path)
        self.recall.load()

    def init_ui(self):

        self.ui = Tk()
        self.ui.resizable(width=False, height=False)
        self.ui.geometry("{}x{}".format(self.UI_WIDTH, self.UI_HEIGHT))

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
        self.h_upper_limit.set(255)
        self.h_upper_limit.grid(row=2, column=0)

        s_upper_limit_label = Label(hsv_filter_frame, text="S Upper Lim", pady=5, padx=5)
        s_upper_limit_label.grid(row=1, column=1)
        s_upper_limit_var = DoubleVar()
        self.s_upper_limit = Scale(hsv_filter_frame, variable=s_upper_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.s_upper_limit.set(255)
        self.s_upper_limit.grid(row=2, column=1)

        v_upper_limit_label = Label(hsv_filter_frame, text="V Upper Lim", pady=5, padx=5)
        v_upper_limit_label.grid(row=1, column=2)
        v_upper_limit_var = DoubleVar()
        self.v_upper_limit = Scale(hsv_filter_frame, variable=v_upper_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.v_upper_limit.set(255)
        self.v_upper_limit.grid(row=2, column=2)

        h_lower_limit_label = Label(hsv_filter_frame, text="H Lower Lim", pady=5, padx=5)
        h_lower_limit_label.grid(row=3, column=0)
        h_lower_limit_var = DoubleVar()
        self.h_lower_limit = Scale(hsv_filter_frame, variable=h_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.h_lower_limit.grid(row=4, column=0)

        s_lower_limit_label = Label(hsv_filter_frame, text="S Lower Lim", pady=5, padx=5)
        s_lower_limit_label.grid(row=3, column=1)
        s_lower_limit_var = DoubleVar()
        self.s_lower_limit = Scale(hsv_filter_frame, variable=s_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.s_lower_limit.grid(row=4, column=1)

        v_lower_limit_label = Label(hsv_filter_frame, text="V Lower Lim", pady=5, padx=5)
        v_lower_limit_label.grid(row=3, column=2)
        v_lower_limit_var = DoubleVar()
        self.v_lower_limit = Scale(hsv_filter_frame, variable=v_lower_limit_var, from_=255,to=0, command=lambda x: self.update_hsv())
        self.v_lower_limit.grid(row=4, column=2)

    def init_vision_controls(self):

        vision_controls_frame = LabelFrame(self.ui, pady=15, padx=15)
        vision_controls_frame.grid(row=11, column=0, rowspan=1, columnspan=10)

        self.toggle_vision = Button(vision_controls_frame, text="Toggle Vision", command=lambda: self.toggle_retina())
        self.toggle_vision.grid(row=0,column=0)

        self.toggle_lane_detection = Button(vision_controls_frame, text="Toggle Lane Detection", command=lambda: print("Enable Lane Detection"))
        self.toggle_lane_detection.grid(row=0, column=1)

        self.toggle_splitter_detection = Button(vision_controls_frame, text="Toggle Splitter Detection", command=lambda: print("Toggle Splitter Detection"))
        self.toggle_splitter_detection.grid(row=0, column=3)

    def init_vision_features(self):

        vision_feature_frame = Frame(self.ui)
        vision_feature_frame.grid(row=12, column=0, rowspan=3, columnspan=10)

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

    def init_img_controls(self):

        img_controls_frame = LabelFrame(self.ui, pady=15, padx=15)
        img_controls_frame.grid(row=10, column=0, rowspan=1, columnspan=10)

        self.next = Button(img_controls_frame, text="Next", command=lambda: self.next_img())
        self.next.grid(row=0, column=0)

        self.previous = Button(img_controls_frame, text="Previous", command=lambda: self.previous_img())
        self.previous.grid(row=0, column=1)

        self.play = Button(img_controls_frame, text="Play", command=lambda: self.start_video())
        self.play.grid(row=0, column=3)

        self.stop = Button(img_controls_frame, text="Stop", command=lambda: self.stop_video())
        self.stop.grid(row=0, column=4)

    def init_canvas(self):

        self.canvas = Canvas(self.ui, width=self.IMG_WIDTH, height=self.IMG_HEIGHT)
        self.canvas.grid(row=0,column=0, rowspan=10, columnspan=10)
        self.canvas_img = self.canvas.create_image((10, 10), image=(), anchor='nw')

    def change_img(self, img_index):

        self.get_image(img_index)

        if self.apply_retina == False:

            self.raw = self.raw[40:80:, :]
            self.img = ImageTk.PhotoImage(self.resize_im(self.raw))

            self.update_img()
            self.logger.info("Image {} opened".format(self.img_index))

        elif self.apply_retina == True:

            self.retina.frame = self.raw
            self.retina.process()
            self.processed = self.retina.frame
            self.img = ImageTk.PhotoImage(self.resize_im(self.processed))

            self.update_img()
            self.logger.info("Image {} opened".format(self.img_index))

    def get_image(self, image_num):

        self.raw = self.recall.frames[image_num]
        self.recall.img_num = image_num
        self.img = ImageTk.PhotoImage(self.resize_im(self.raw))

    def update_img(self):

        self.canvas.itemconfigure(self.canvas_img, image=self.img)

    def resize_im(self, im):



        im = self.recall.rgb_to_img(im)
        return im.resize((128 * self.RESIZE_FACTOR, 40 * self.RESIZE_FACTOR), PIL.Image.NEAREST)
        # return im

    def next_img(self):

        if self.img_index == self.recall.num_images - 1:
            self.logger.info("Last image reached")
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
            if self.img_index >= self.recall.num_images:
                self.img_index = 0

            self.change_img(self.img_index)

            time.sleep(0.25)

    def update_hsv(self):

        self.retina.fil_hsv_u[0] = int(self.h_upper_limit.get())
        self.retina.fil_hsv_u[1] = int(self.s_upper_limit.get())
        self.retina.fil_hsv_u[2] = int(self.v_upper_limit.get())

        self.retina.fil_hsv_l[0] = int(self.h_lower_limit.get())
        self.retina.fil_hsv_l[1] = int(self.s_lower_limit.get())
        self.retina.fil_hsv_l[2] = int(self.v_lower_limit.get())

        # self.retina.set_calibration(self.TYPE, self.retina.fil_rgb_l, self.retina.fil_rgb_u)

        self.change_img(self.img_index)

    def toggle_retina(self):

        if not self.apply_retina:
            self.apply_retina = True

        elif self.apply_retina:
            self.apply_retina = False

        self.change_img(self.img_index)

    def run(self):

        self.ui.mainloop()

if __name__ == '__main__':

    print("Platform: {}".format(platform.platform()))

    if 'Darwin' in platform.platform():
        data_path = "/Users/arnavgupta/car_data/raw_npy/oculus-2019-06-29 18;29;43.996328.npy"
    else:
        data_path = r"/home/veda/git/auto-rc_poc/autorc/data/oculus-2019-06-29 18;29;43.996328.npy"


    simulator = Simulator(data_path)
    simulator.run()