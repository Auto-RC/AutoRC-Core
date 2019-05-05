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

    UI_HEIGHT = 400
    UI_WIDTH = 600

    IMG_WIDTH = 200
    IMG_HEIGHT = 200

    # -------------------------- Initialization --------------------------------

    def __init__(self):

        self.init_ui()

        self.init_recall()
        self.init_retina()

        self.init_img_ctrls()

        self.img_index = 0
        self.get_image(self.img_index)
        self.update_img()

        self.video_active = False

        self.apply_retina = True

    def init_ui(self):

        self.ui = tk.Tk()
        self.ui.resizable(width=False,height=False)
        self.ui.geometry("{}x{}".format(self.UI_WIDTH,self.UI_HEIGHT))

        self.canvas = tk.Canvas(self.ui,width=self.IMG_WIDTH,height=self.IMG_HEIGHT)
        self.canvas.place(x=10,y=10)

        self.canvas_img = self.canvas.create_image((10,10), image=(), anchor='nw')

    def init_img_ctrls(self):

        self.next = tk.Button(self.ui, text="  Next  ", command=lambda:self.next_img())
        self.next.config(width=8)
        self.next.place(x=20,y=230)

        self.previous = tk.Button(self.ui, text="Previous", command=lambda:self.previous_img())
        self.previous.config(width=8)
        self.previous.place(x=120, y=230)

        self.start = tk.Button(self.ui, text="Play",command=lambda: self.start_video())
        self.start.config(width=8)
        self.start.place(x=20, y=260)

        self.stop = tk.Button(self.ui, text="Stop",command=lambda: self.stop_video())
        self.stop.config(width=8)
        self.stop.place(x=120, y=260)

    def init_vision_ctrls(self):

        self.recalibrate = tk.Button(self.ui, text="Recalibrate", command=lambda:print("Run button pressed"))
        self.recalibrate.config(width=9)
        self.recalibrate.place(x=475, y=80)

    def init_recall(self):

        self.recall = Recall(r"/media/sf_VM_Shared/autorc_data/oculus-2019-04-20 17;48;15.783634.npy")
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

            self.update_img()
            logger.info("Image {} opened (Retina applied: {})".format(self.img_index,self.apply_retina))

        elif self.apply_retina == True:

            self.retina.frame = self.raw
            self.processed = self.retina.process()
            self.img = ImageTk.PhotoImage(self.recall.rgb_to_img(self.processed))

            self.update_img()
            logger.info("Image {} opened (Retina applied: {})".format(self.img_index,self.apply_retina))

    # -------------------------- Image Utilities -------------------------------

    def get_image(self, image_num):

        self.raw = self.recall.frames[image_num]
        self.img = ImageTk.PhotoImage(self.recall.rgb_to_img(self.raw))

    def update_img(self):

        self.canvas.itemconfigure(self.canvas_img,image=self.img)

    # ---------------------------- GUI Startup ---------------------------------

    def run(self):

        self.ui.mainloop()

# ------------------------------------------------------------------------------
#                                  ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    calibrator = Calibrator()
    calibrator.run()

