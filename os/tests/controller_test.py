from controller import Controller

controller = Controller()

while(1):
    controller.update()
    print(controller.steering, controller.throttle)