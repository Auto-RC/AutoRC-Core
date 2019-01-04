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

safety = True
brake_safety = True

cam = PiCamera()
controller = Controller()
drive = Car_Control(PWM_FREQ)
memory = Memory()

t = Thread(target=cam.update, args=())
t.daemon = True
t.start()
time.sleep(1)
print("started thread")


try:
    i = 0
    while(1):
        controller.update()
        print("controller updated", i)


        if controller.throttle != 0:
            safety = False
        if controller.throttle == 0 and safety:
            controller.throttle = -1

        if controller.brake != 0:
            brake_safety = False
        if controller.brake == 0 and brake_safety:
            controller.brake = -1

        throttle = calcThrottle(controller.throttle, controller.brake)

        print("throttle: ", throttle, "steering: ", controller.steering, i)

        drive.set_steering(controller.steering)
        drive.set_throttle(throttle)

        frame = cam.run_threaded()

        memory.add(frame, [controller.steering, throttle])
        print("memory appended", i)
        i +=1
except Exception as e:
    cam.shutdown()
    memory.save()
    print(e)