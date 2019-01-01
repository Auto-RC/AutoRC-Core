from controller import Controller
from carControl import Car_Control

PWM_FREQ = 60

controller = Controller()
drive = Car_Control(PWM_FREQ)


while(1):
    controller.update()
    drive.set_steering(controller.steering)
