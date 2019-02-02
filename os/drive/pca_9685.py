from __future__ import division
import time

import Adafruit_PCA9685


class PCA9685:

    def __init__(self, freq=60):

        self.pwm = Adafruit_PCA9685.PCA9685()
        self.steering_min = 300
        self.steering_max = 490
        self.steering_avg = self.steering_min/2 + self.steering_max/2
        self.throttle_min = 290
        self.throttle_max = 410
        self.throttle_avg = 390
        self.pwm.set_pwm_freq(freq)


    def set_steering(self, position):

        if -1 <= position <= 1:
            position = int((-position * (self.steering_max-self.steering_avg)) + self.steering_avg)
        if position > self.steering_max:
            self.pwm.set_pwm(0, 0, self.steering_max)
        elif position < self.steering_min:
            self.pwm.set_pwm(0, 0, self.steering_min)
        else:
            self.pwm.set_pwm(0, 0, position)

    def set_throttle(self, throttle):
        if -1 <= throttle < 0:
            throttle = int(throttle * (self.throttle_avg-self.throttle_min)) + self.throttle_avg
        elif 0 <= throttle <= 1:
            throttle = int(throttle * (self.throttle_max-self.throttle_avg)) + self.throttle_avg
        else:
            throttle = 0
        self.pwm.set_pwm(1, 0, throttle)
