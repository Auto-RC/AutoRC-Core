# ------------------------------------------------------------------------------
#                               GLOBAL IMPORTS
# ------------------------------------------------------------------------------

import os
import logging
import tkinter as tk
from PIL import Image,ImageTk
import threading
import time
import numpy as np

# ------------------------------------------------------------------------------
#                               LOCAL IMPORTS
# ------------------------------------------------------------------------------

from recall import Recall
from retina import Retina

# ------------------------------------------------------------------------------
#                                SETUP LOGGING
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------
#                                  CALIBRATOR
# ------------------------------------------------------------------------------

class Calibrator(threading.Thread):

    UI_HEIGHT = 350
    UI_WIDTH = 750

    IMG_WIDTH = 400
    IMG_HEIGHT = 200

    INITAL_IMG_INDEX = 155

    ADJ_MAG = 5

    TYPE = 'lane' # 'splitter'

    # -------------------------- Initialization --------------------------------

    def __init__(self):

        self.video_active = False
        self.apply_retina = True

        self.init_ui()

        self.init_recall()
        self.init_retina()

        self.init_img_ctrls()

        self.img_index = self.INITAL_IMG_INDEX
        self.change_img(self.img_index)
        self.update_img()

    def init_ui(self):

        self.ui = tk.Tk()
        self.ui.resizable(width=False,height=False)
        self.ui.geometry("{}x{}".format(self.UI_WIDTH,self.UI_HEIGHT))

        self.canvas = tk.Canvas(self.ui,width=self.IMG_WIDTH,height=self.IMG_HEIGHT)
        self.canvas.place(x=10,y=10)

        self.canvas_img = self.canvas.create_image((10,10), image=(), anchor='nw')

        self.canvas.bind("<Button 1>", self.mouse_click_event)

    def init_img_ctrls(self):

        self.next = tk.Button(self.ui, text="  Next  ", command=lambda:self.next_img())
        self.next.config(width=8)
        self.next.place(x=15,y=280)

        self.previous = tk.Button(self.ui, text="Previous", command=lambda:self.previous_img())
        self.previous.config(width=8)
        self.previous.place(x=120, y=280)

        self.start = tk.Button(self.ui, text="Play",command=lambda: self.start_video())
        self.start.config(width=8)
        self.start.place(x=15, y=310)

        self.stop = tk.Button(self.ui, text="Stop",command=lambda: self.stop_video())
        self.stop.config(width=8)
        self.stop.place(x=120, y=310)

        self.fil_l_rgb1_increase = tk.Button(self.ui, text="Inc R L Fil", command=lambda: self.adjust_lower_fil(index=0,vector="increase", mag=self.ADJ_MAG))
        self.fil_l_rgb1_increase.config(width=8)
        self.fil_l_rgb1_increase.place(x=320, y=20)

        self.fil_l_rgb2_increase = tk.Button(self.ui, text="Inc G L Fil",command=lambda: self.adjust_lower_fil(index=1, vector="increase",mag=self.ADJ_MAG))
        self.fil_l_rgb2_increase.config(width=8)
        self.fil_l_rgb2_increase.place(x=320, y=50)

        self.fil_l_rgb3_increase = tk.Button(self.ui, text="Inc B L Fil",command=lambda: self.adjust_lower_fil(index=2, vector="increase",mag=self.ADJ_MAG))
        self.fil_l_rgb3_increase.config(width=8)
        self.fil_l_rgb3_increase.place(x=320, y=80)

        self.fil_l_rgb1_decrease = tk.Button(self.ui, text="Dec R L Fil",command=lambda: self.adjust_lower_fil(index=0, vector="decrease",mag=self.ADJ_MAG))
        self.fil_l_rgb1_decrease.config(width=8)
        self.fil_l_rgb1_decrease.place(x=420, y=20)

        self.fil_l_rgb2_decrease = tk.Button(self.ui, text="Dec G L Fil",command=lambda: self.adjust_lower_fil(index=1, vector="decrease",mag=self.ADJ_MAG))
        self.fil_l_rgb2_decrease.config(width=8)
        self.fil_l_rgb2_decrease.place(x=420, y=50)

        self.fil_l_rgb3_decrease = tk.Button(self.ui, text="Dec B L Fil",command=lambda: self.adjust_lower_fil(index=2, vector="decrease",mag=self.ADJ_MAG))
        self.fil_l_rgb3_decrease.config(width=8)
        self.fil_l_rgb3_decrease.place(x=420, y=80)

        self.fil_u_rgb1_increase = tk.Button(self.ui, text="Inc R U Fil", command=lambda: self.adjust_upper_fil(index=0, vector="increase", mag=self.ADJ_MAG))
        self.fil_u_rgb1_increase.config(width=8)
        self.fil_u_rgb1_increase.place(x=320, y=120)

        self.fil_u_rgb2_increase = tk.Button(self.ui, text="Inc G U Fil", command=lambda: self.adjust_upper_fil(index=1, vector="increase", mag=self.ADJ_MAG))
        self.fil_u_rgb2_increase.config(width=8)
        self.fil_u_rgb2_increase.place(x=320, y=150)

        self.fil_u_rgb3_increase = tk.Button(self.ui, text="Inc B U Fil", command=lambda: self.adjust_upper_fil(index=2, vector="increase", mag=self.ADJ_MAG))
        self.fil_u_rgb3_increase.config(width=8)
        self.fil_u_rgb3_increase.place(x=320, y=180)

        self.fil_u_rgb1_decrease = tk.Button(self.ui, text="Dec R U Fil", command=lambda: self.adjust_upper_fil(index=0, vector="decrease", mag=self.ADJ_MAG))
        self.fil_u_rgb1_decrease.config(width=8)
        self.fil_u_rgb1_decrease.place(x=420, y=120)

        self.fil_u_rgb2_decrease = tk.Button(self.ui, text="Dec G U Fil", command=lambda: self.adjust_upper_fil(index=1, vector="decrease", mag=self.ADJ_MAG))
        self.fil_u_rgb2_decrease.config(width=8)
        self.fil_u_rgb2_decrease.place(x=420, y=150)

        self.fil_u_rgb3_decrease = tk.Button(self.ui, text="Dec B U Fil", command=lambda: self.adjust_upper_fil(index=2, vector="decrease", mag=self.ADJ_MAG))
        self.fil_u_rgb3_decrease.config(width=8)
        self.fil_u_rgb3_decrease.place(x=420, y=180)

        self.fil_l_hsv1_increase = tk.Button(self.ui, text="Inc H L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=0,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv1_increase.config(width=8)
        self.fil_l_hsv1_increase.place(x=520, y=20)

        self.fil_l_hsv2_increase = tk.Button(self.ui, text="Inc S L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=1,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv2_increase.config(width=8)
        self.fil_l_hsv2_increase.place(x=520, y=50)

        self.fil_l_hsv3_increase = tk.Button(self.ui, text="Inc V L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=2,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv3_increase.config(width=8)
        self.fil_l_hsv3_increase.place(x=520, y=80)

        self.fil_l_hsv1_decrease = tk.Button(self.ui, text="Dec H L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=0,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv1_decrease.config(width=8)
        self.fil_l_hsv1_decrease.place(x=620, y=20)

        self.fil_l_hsv2_decrease = tk.Button(self.ui, text="Dec S L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=1,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv2_decrease.config(width=8)
        self.fil_l_hsv2_decrease.place(x=620, y=50)

        self.fil_l_hsv3_decrease = tk.Button(self.ui, text="Dec V L Fil",
                                             command=lambda: self.adjust_lower_hsv_fil(index=2,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_l_hsv3_decrease.config(width=8)
        self.fil_l_hsv3_decrease.place(x=620, y=80)

        self.fil_u_hsv1_increase = tk.Button(self.ui, text="Inc H U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=0,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv1_increase.config(width=8)
        self.fil_u_hsv1_increase.place(x=520, y=120)

        self.fil_u_hsv2_increase = tk.Button(self.ui, text="Inc S U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=1,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv2_increase.config(width=8)
        self.fil_u_hsv2_increase.place(x=520, y=150)

        self.fil_u_hsv3_increase = tk.Button(self.ui, text="Inc V U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=2,
                                                                                   vector="increase",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv3_increase.config(width=8)
        self.fil_u_hsv3_increase.place(x=520, y=180)

        self.fil_u_hsv1_decrease = tk.Button(self.ui, text="Dec H U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=0,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv1_decrease.config(width=8)
        self.fil_u_hsv1_decrease.place(x=620, y=120)

        self.fil_u_hsv2_decrease = tk.Button(self.ui, text="Dec S U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=1,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv2_decrease.config(width=8)
        self.fil_u_hsv2_decrease.place(x=620, y=150)

        self.fil_u_hsv3_decrease = tk.Button(self.ui, text="Dec V U Fil",
                                             command=lambda: self.adjust_upper_hsv_fil(index=2,
                                                                                   vector="decrease",
                                                                                   mag=self.ADJ_MAG))
        self.fil_u_hsv3_decrease.config(width=8)
        self.fil_u_hsv3_decrease.place(x=620, y=180)

        self.display_lanes = tk.Button(self.ui, text="Display Lanes",command=lambda: self.enable_lanes())
        self.display_lanes.config(width=8)
        self.display_lanes.place(x=320, y=220)

        self.remove_lanes = tk.Button(self.ui, text="Remove Lanes",command=lambda: self.disable_lanes())
        self.remove_lanes.config(width=8)
        self.remove_lanes.place(x=420, y=220)

        self.enable_vision = tk.Button(self.ui, text="Enable Vision", command=lambda: self.enable_retina())
        self.enable_vision.config(width=8)
        self.enable_vision.place(x=320, y=260)

        self.disable_vision = tk.Button(self.ui, text="Disable Vision", command=lambda: self.disable_retina())
        self.disable_vision.config(width=8)
        self.disable_vision.place(x=420, y=260)

        self.show_rgb = tk.Button(self.ui, text="RGB", command=lambda: self.enable_RGB())
        self.show_rgb.config(width=8)
        self.show_rgb.place(x=320, y=300)

        self.show_hsv = tk.Button(self.ui, text="HSV", command=lambda: self.enable_HSV())
        self.show_hsv.config(width=8)
        self.show_hsv.place(x=420, y=300)

    def init_vision_ctrls(self):

        self.recalibrate = tk.Button(self.ui, text="Recalibrate", command=lambda:print("Run button pressed"))
        self.recalibrate.config(width=9)
        self.recalibrate.place(x=475, y=80)

    def init_recall(self):

        self.recall = Recall("/Users/arnavgupta/car_data/raw_npy/oculus-2019-06-26 10;51;31.605013.npy")
        self.recall.load()

    # ------------------------- Retina Integration -----------------------------

    def init_retina(self):

        self.retina = Retina()

    # ---------------------------- Video Thread --------------------------------

    def start_video(self):

        self.video_active = True
        self.video = threading.Thread(target=self.video_thread,args=())
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

    # ----------------------------- Calibration --------------------------------

    def adjust_lower_fil(self, index, vector="increase", mag=5):

        if vector == "increase":
            if self.retina.fil_rgb_l[index] < 255:
                self.retina.fil_rgb_l[index] += mag
        elif vector == "decrease":
            if self.retina.fil_rgb_l[index] > 0:
                self.retina.fil_rgb_l[index] -= mag

        self.retina.set_calibration(self.TYPE, self.retina.fil_rgb_l, self.retina.fil_rgb_u)

        logger.info("Lower RGB Filter: {} Upper RGB Filter: {} Lower HSV Filter: {} Upper HSV Filter: {}".format(self.retina.fil_rgb_l,self.retina.fil_rgb_u, self.retina.fil_hsv_l, self.retina.fil_hsv_u))

        self.change_img(self.img_index)

    def adjust_upper_fil(self, index, vector="increase", mag=5):

        if vector == "increase":
            if self.retina.fil_rgb_u[index] < 255:
                self.retina.fil_rgb_u[index] += mag
        elif vector == "decrease":
            if self.retina.fil_rgb_u[index] > 0:
                self.retina.fil_rgb_u[index] -= mag

        self.retina.set_calibration(self.TYPE, self.retina.fil_rgb_l, self.retina.fil_rgb_u)

        logger.info("Lower RGB Filter: {} Upper RGB Filter: {} Lower HSV Filter: {} Upper HSV Filter: {}".format(self.retina.fil_rgb_l,self.retina.fil_rgb_u, self.retina.fil_hsv_l, self.retina.fil_hsv_u))

        self.change_img(self.img_index)

    def adjust_lower_hsv_fil(self, index, vector="increase", mag=5):

        if vector == "increase":
            if self.retina.fil_hsv_l[index] < 255:
                self.retina.fil_hsv_l[index] += mag
        elif vector == "decrease":
            if self.retina.fil_hsv_l[index] > 0:
                self.retina.fil_hsv_l[index] -= mag

        self.retina.set_calibration(self.TYPE, self.retina.fil_hsv_l, self.retina.fil_hsv_u)

        logger.info("Lower RGB Filter: {} Upper RGB Filter: {} Lower HSV Filter: {} Upper HSV Filter: {}".format(self.retina.fil_rgb_l,self.retina.fil_rgb_u, self.retina.fil_hsv_l, self.retina.fil_hsv_u))

        self.change_img(self.img_index)

    def adjust_upper_hsv_fil(self, index, vector="increase", mag=5):

        if vector == "increase":
            if self.retina.fil_hsv_u[index] < 255:
                self.retina.fil_hsv_u[index] += mag
        elif vector == "decrease":
            if self.retina.fil_hsv_u[index] > 0:
                self.retina.fil_hsv_u[index] -= mag

        self.retina.set_calibration(self.TYPE, self.retina.fil_hsv_l, self.retina.fil_hsv_u)

        logger.info("Lower RGB Filter: {} Upper RGB Filter: {} Lower HSV Filter: {} Upper HSV Filter: {}".format(self.retina.fil_rgb_l,self.retina.fil_rgb_u, self.retina.fil_hsv_l, self.retina.fil_hsv_u))

        self.change_img(self.img_index)

    def enable_lanes(self):

        self.retina.enable_lines = True
        self.change_img(self.img_index)

    def disable_lanes(self):

        self.retina.enable_lines = False
        self.change_img(self.img_index)

    def enable_retina(self):

        self.apply_retina = True
        self.change_img(self.img_index)

    def disable_retina(self):

        self.apply_retina = False
        self.change_img(self.img_index)

    def enable_RGB(self):

        self.retina.mode = 'RGB'
        self.change_img(self.img_index)

    def enable_HSV(self):

        self.retina.mode = 'HSV'
        self.change_img(self.img_index)

    # ---------------------------- Image Change --------------------------------

    def next_img(self):

        if self.img_index == self.recall.num_images-1:
            logger.info("Last image reached")
        else:
            self.img_index += 1
            self.change_img(self.img_index)

    def previous_img(self):

        if self.img_index == 0:
            logger.info("First image reached")
        else:
            self.img_index -= 1
            self.change_img(self.img_index)

    def change_img(self, img_index):

        self.get_image(img_index)

        if self.apply_retina == False:

            self.raw = self.raw[40:80:, :]
            self.img = ImageTk.PhotoImage(self.resize_im(self.raw))
            self.update_img()
            logger.info("Image {} opened (Retina applied: {})".format(self.img_index,self.apply_retina))

        elif self.apply_retina == True:

            self.retina.frame = self.raw
            theta, rho = self.retina.process()
            self.processed = self.retina.frame
            print(theta, rho)
            self.img = ImageTk.PhotoImage(self.resize_im(self.processed))

            self.update_img()
            logger.info("Image {} opened (Retina applied: {})".format(self.img_index,self.apply_retina))



    # -------------------------- Image Utilities -------------------------------

    def get_image(self, image_num):

        self.raw = self.recall.frames[image_num]
        self.img = ImageTk.PhotoImage(self.resize_im(self.raw))


    def update_img(self):

        self.canvas.itemconfigure(self.canvas_img, image=self.img)

    def resize_im(self, im):
        im = self.recall.rgb_to_img(im)
        return im.resize((128 * 2, 40 * 2), Image.NEAREST)
        # return im

    # ---------------------------- GUI Startup ---------------------------------

    def run(self):

        self.ui.mainloop()

    # ---------------------------- Pixel Info ----------------------------------

    def mouse_click_event(self, event_origin):

        global x0, y0
        x0 = event_origin.x
        y0 = event_origin.y
        print("Pixel: ({},{}) Shape:({}) RGB: ({})".format(x0,y0,self.raw.shape,self.raw[y0][x0]))

# ------------------------------------------------------------------------------
#                                  ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    calibrator = Calibrator()
    calibrator.run()

