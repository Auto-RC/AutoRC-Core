from controller import Controller
from carControl import Car_Control
import time

PWM_FREQ = 60

controller = Controller()
drive = Car_Control(PWM_FREQ)

safety = True
brake_safety = True

def calcThrottle(throttle, brake):
    if brake > -0.8:
        return -(brake+1)/2
    else:
        return (throttle+1)/2

while(1):
    s = time.time()
    controller.update()

    if controller.throttle != 0:
        safety = False
    if controller.throttle == 0 and safety:
        controller.throttle = -1

    if controller.brake != 0:
        brake_safety = False
    if controller.brake == 0 and brake_safety:
        controller.brake = -1

    print(controller.steering, controller.throttle, controller.brake)
    drive.set_steering(controller.steering)
    drive.set_throttle(calcThrottle(controller.throttle, controller.brake))
    print("total time:", time.time() - s)