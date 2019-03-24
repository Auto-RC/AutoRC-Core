
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Constant
#define THR_CH  1
#define STR_CH  0

// Receiver calibration
float STEERINGMIN = 270; // this is the 'minimum' pulse length count (out of 4096)
float STEERINGMAX = 430; // this is the 'maximum' pulse length count (out of 4096)
float THROTTLEMIN = 348;
float THROTTLEMAX = 360;
float THRINMAX = 1925;
float THRINMIN = 1000;
float STRINMAX = 1990;
float STRINMIN = 1050;

// These hold the input from the receiver
int thr_input;
int str_input;
int swb_input;
int swc_input;

// Hold the raw control values
float thr = 0;
float str = 0;

// These are sent to the PCA
int thr_out = 0;
int str_out = 0;

// Controls the PCA
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setup()
{
    Serial.begin(9600);

    pwm.begin();

    pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates

    pinMode(2,INPUT);
    pinMode(4,INPUT);
    pinMode(6,INPUT);
    pinMode(7,INPUT);

    delay(10);
}

void loop() 
{
    thr_input = pulseIn(4,HIGH,1000000);
    str_input = pulseIn(2,HIGH,1000000);
    swb_input = pulseIn(6,HIGH,1000000);
    swc_input = pulseIn(7,HIGH,1000000);

    if (swb_input > 1500)
    {
        thr = ( (thr_input-THRINMIN)/(THRINMAX-THRINMIN) )*(THROTTLEMAX-THROTTLEMIN)+THROTTLEMIN;
        thr_out = thr_output/1;

        str_output = ( (str_input-STRINMIN)/(STRINMAX-STRINMIN) )*(STEERINGMAX-STEERINGMIN)+STEERINGMIN;
        str_out = str_output/1;

        pwm.setPWM(THR_CH, 0, thr_out);
        pwm.setPWM(STR_CH, 0, str_out);
    }
    else
    {
        str_out = (STEERINGMAX-STEERINGMIN)/2;

        pwm.setPWM(THR_CH, 0, THROTTLEMIN);
        pwm.setPWM(STR_CH, 0, str_out);
    }
}
