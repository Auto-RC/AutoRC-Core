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
from autorc.vehicle.vision.recall import Recall

class Simulator():

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

        # Init UI
        self.init_ui()

        # Init RGB Controls
        self.init_vision_rgb_controls()

        # Init HVS Controls
        self.init_vision_hsv_controls()

        # Init Vision Features
        self.init_vision_controls()
        self.init_vision_features()

        # Init recall
        self.init_recall(data_path)

        # Init Canvas
        self.init_canvas()
        self.img_index = 24
        self.change_img(self.img_index)
        self.update_img()

    def init_recall(self, data_path):

        self.recall = Recall(data_path)
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
        r_upper_limit_var = DoubleVar()
        r_upper_limit = Scale(rgb_filter_frame, variable=r_upper_limit_var)
        r_upper_limit.grid(row=2, column=0)

        g_upper_limit_label = Label(rgb_filter_frame, text="G Upper Lim", pady=5, padx=5)
        g_upper_limit_label.grid(row=1, column=1)
        g_upper_limit_var = DoubleVar()
        g_upper_limit = Scale(rgb_filter_frame, variable=g_upper_limit_var)
        g_upper_limit.grid(row=2, column=1)

        b_upper_limit_label = Label(rgb_filter_frame, text="B Upper Lim", pady=5, padx=5)
        b_upper_limit_label.grid(row=1, column=2)
        b_upper_limit_var = DoubleVar()
        b_upper_limit = Scale(rgb_filter_frame, variable=b_upper_limit_var)
        b_upper_limit.grid(row=2, column=2)

        r_lower_limit_label = Label(rgb_filter_frame, text="R Lower Lim", pady=5, padx=5)
        r_lower_limit_label.grid(row=3, column=0)
        r_lower_limit_var = DoubleVar()
        r_lower_limit = Scale(rgb_filter_frame, variable=r_lower_limit_var)
        r_lower_limit.grid(row=4, column=0)

        g_lower_limit_label = Label(rgb_filter_frame, text="G Lower Lim", pady=5, padx=5)
        g_lower_limit_label.grid(row=3, column=1)
        g_lower_limit_var = DoubleVar()
        g_lower_limit = Scale(rgb_filter_frame, variable=g_lower_limit_var)
        g_lower_limit.grid(row=4, column=1)

        b_lower_limit_label = Label(rgb_filter_frame, text="B Lower Lim", pady=5, padx=5)
        b_lower_limit_label.grid(row=3, column=2)
        b_lower_limit_var = DoubleVar()
        b_lower_limit = Scale(rgb_filter_frame, variable=b_lower_limit_var)
        b_lower_limit.grid(row=4, column=2)

    def init_vision_hsv_controls(self):

        hsv_filter_frame = LabelFrame(self.ui, text="HSV Filters", pady=15, padx=15)
        hsv_filter_frame.grid(row=10, column=10, rowspan=10, columnspan=15)

        h_upper_limit_label = Label(hsv_filter_frame, text="H Upper Lim", pady=5, padx=5)
        h_upper_limit_label.grid(row=1, column=0)
        h_upper_limit_var = DoubleVar()
        h_upper_limit = Scale(hsv_filter_frame, variable=h_upper_limit_var)
        h_upper_limit.grid(row=2, column=0)

        s_upper_limit_label = Label(hsv_filter_frame, text="S Upper Lim", pady=5, padx=5)
        s_upper_limit_label.grid(row=1, column=1)
        s_upper_limit_var = DoubleVar()
        s_upper_limit = Scale(hsv_filter_frame, variable=s_upper_limit_var)
        s_upper_limit.grid(row=2, column=1)

        v_upper_limit_label = Label(hsv_filter_frame, text="V Upper Lim", pady=5, padx=5)
        v_upper_limit_label.grid(row=1, column=2)
        v_upper_limit_var = DoubleVar()
        v_upper_limit = Scale(hsv_filter_frame, variable=v_upper_limit_var)
        v_upper_limit.grid(row=2, column=2)

        h_lower_limit_label = Label(hsv_filter_frame, text="H Lower Lim", pady=5, padx=5)
        h_lower_limit_label.grid(row=3, column=0)
        h_lower_limit_var = DoubleVar()
        h_lower_limit = Scale(hsv_filter_frame, variable=h_lower_limit_var)
        h_lower_limit.grid(row=4, column=0)

        s_lower_limit_label = Label(hsv_filter_frame, text="S Lower Lim", pady=5, padx=5)
        s_lower_limit_label.grid(row=3, column=1)
        s_lower_limit_var = DoubleVar()
        s_lower_limit = Scale(hsv_filter_frame, variable=s_lower_limit_var)
        s_lower_limit.grid(row=4, column=1)

        v_lower_limit_label = Label(hsv_filter_frame, text="V Lower Lim", pady=5, padx=5)
        v_lower_limit_label.grid(row=3, column=2)
        v_lower_limit_var = DoubleVar()
        v_lower_limit = Scale(hsv_filter_frame, variable=v_lower_limit_var)
        v_lower_limit.grid(row=4, column=2)

    def init_vision_controls(self):

        vision_controls_frame = Frame(self.ui, pady=15, padx=15)
        vision_controls_frame.grid(row=10, column=0, rowspan=1, columnspan=10)

        self.toggle_vision = Button(vision_controls_frame, text="Toggle Vision", command=lambda: print("Toggle Vision"))
        self.toggle_vision.grid(row=0,column=0)

        self.toggle_lane_detection = Button(vision_controls_frame, text="Toggle Lane Detection", command=lambda: print("Enable Lane Detection"))
        self.toggle_lane_detection.grid(row=0, column=1)

        self.toggle_splitter_detection = Button(vision_controls_frame, text="Toggle Splitter Detection", command=lambda: print("Toggle Splitter Detection"))
        self.toggle_splitter_detection.grid(row=0, column=3)

    def init_vision_features(self):

        vision_feature_frame = Frame(self.ui)
        vision_feature_frame.grid(row=11, column=0, rowspan=3, columnspan=10)

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

    def init_canvas(self):

        self.canvas = Canvas(self.ui, width=self.IMG_WIDTH, height=self.IMG_HEIGHT)
        self.canvas.grid(row=0,column=0, rowspan=10, columnspan=10)
        self.canvas_img = self.canvas.create_image((10, 10), image=(), anchor='nw')

    def change_img(self, img_index):

        self.get_image(img_index)

        self.raw = self.raw[40:80:, :]
        self.img = ImageTk.PhotoImage(self.resize_im(self.raw))
        self.update_img()
        self.logger.info("Image {} opened".format(self.img_index))

    def get_image(self, image_num):

        self.raw = self.recall.frames[image_num]
        self.img = ImageTk.PhotoImage(self.resize_im(self.raw))

    def update_img(self):

        self.canvas.itemconfigure(self.canvas_img, image=self.img)

    def resize_im(self, im):



        im = self.recall.rgb_to_img(im)
        return im.resize((128 * self.RESIZE_FACTOR, 40 * self.RESIZE_FACTOR), PIL.Image.NEAREST)
        # return im

    def run(self):

        self.ui.mainloop()

if __name__ == '__main__':

    data_path = r"/home/veda/git/AutoRC-Core/autorc/data/oculus-2019-06-29 18;29;43.996328.npy"

    simulator = Simulator(data_path)
    simulator.run()