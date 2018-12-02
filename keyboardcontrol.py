import keyboard
import pygame
import serial
import time

ser = serial.Serial('/dev/cu.usbserial-AH06VWZ6', 250000, timeout=1)
print(ser.isOpen())



forward = 500
side = 500

def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

def disp_vars(forward, side):
    screen.fill((255, 255, 255))
    largeText = pygame.font.Font('freesansbold.ttf', 50)
    TextSurf, TextRect = text_objects(str(forward), largeText)
    TextRect.center = ((240 / 2), (180 / 4))
    screen.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects(str(side), largeText)
    TextRect.center = ((240 / 2), (3 * 180 / 4))
    screen.blit(TextSurf, TextRect)
    send(forward, side)

def send(forward, side):
    for i in range(4 - len(str(side))):
        ser.write(b'0')
    ser.write(str(side).encode())
    for i in range(4 - len(str(forward))):
        ser.write(b'0')
    ser.write(str(forward).encode())
    time.sleep(0.05)

def for_back(forward, max_val = 1110, min_val = 700, avg_val = 900, incr = 5):
    if keyboard.is_pressed('up'):
        if forward < avg_val:
            forward = avg_val
        if forward < max_val:
            forward += incr
    elif keyboard.is_pressed('down'):
        if forward > avg_val:
            forward = avg_val
        if forward > min_val:
            forward -= incr
    else:
        forward = avg_val
    return forward

def lateral(side, max_val = 1110, min_val = 700, avg_val = 900, incr = 15):
    if keyboard.is_pressed('left'):
        if side < avg_val:
            side = avg_val
        if side < max_val:
            side += incr
    elif keyboard.is_pressed('right'):
        if side > avg_val:
            side = avg_val
        if side > min_val:
            side -= incr
    else:
        side = avg_val
    return side

pygame.init()
screen = pygame.display.set_mode((240, 180))
screen.fill((255, 255, 255))

running = True
while running:
    if not keyboard.is_pressed('space'):
        forward = for_back(forward)
        side = lateral(side)
    disp_vars(forward, side)
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

