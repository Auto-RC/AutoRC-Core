from __future__ import division
import time

import Adafruit_PCA9685


class Car_Control:
    def __init__(self, freq=60):
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.steering_min = 289
        self.steering_max = 492
        self.pwm.set_pwm_freq(freq)

    def set_steering(self, position):
        if position > self.steering_max:
            self.pwm.set_pwm(0, 0, self.steering_max)
        elif position > self.steering_min:
            self.pwm.set_pwm(0, 0, self.steering_min)
        else:
            self.pwm.set_pwm(0, 0, position)
