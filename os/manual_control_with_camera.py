from camera import PiCamera
from controller import Controller
from carControl import Car_Control
from threading import Thread
from memory import Memory
import time


def calcThrottle(throttle, brake):
    if brake > -0.8:
        return -(brake+1)/2
    else:
        return (throttle+1)/2

PWM_FREQ = 60
FREQ = 20

safety = True
brake_safety = True

cam = PiCamera()
controller = Controller()
drive = Car_Control(PWM_FREQ)
memory = Memory()

t = Thread(target=cam.update, args=())
t.daemon = True
t.start()

print("started thread")
time.sleep(1)
print("running")

data = 0
i = 0

while controller.on:
    controller.update()

    if controller.throttle != 0:
        safety = False
    if controller.throttle == 0 and safety:
        controller.throttle = -1

    if controller.brake != 0:
        brake_safety = False
    if controller.brake == 0 and brake_safety:
        controller.brake = -1

    throttle = calcThrottle(controller.throttle, controller.brake)

    drive.set_steering(controller.steering)
    drive.set_throttle(throttle)

    if controller.capturing:
        if data == 0:
            i += 1
            print("dataset {}".format(i))
            data = 1

        frame = cam.run_threaded()

        memory.add(frame, [controller.steering, throttle])

    elif data == 1:
        memory.save(str("data{}".format(i)))
        data = 0

    time.sleep(1/FREQ)

print("shutting down")
cam.shutdown()

memory.save(("data" + str(i)))
print("done")