/*************************************************** 
  This is an example for our Adafruit 16-channel PWM & Servo driver
  Servo test - this will drive 8 servos, one after the other on the
  first 8 pins of the PCA9685

  Pick one up today in the adafruit shop!
  ------> http://www.adafruit.com/products/815
  
  These drivers use I2C to communicate, 2 pins are required to  
  interface.

  Adafruit invests time and resources providing this open source code, 
  please support Adafruit and open-source hardware by purchasing 
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);
// you can also call it with a different address and I2C interface
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(&Wire, 0x40);

// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
float STEERINGMIN = 270; // this is the 'minimum' pulse length count (out of 4096)
float STEERINGMAX = 430; // this is the 'maximum' pulse length count (out of 4096)
float THROTTLEMIN = 348;
float THROTTLEMAX = 360;
float THRINMAX = 1925;
float THRINMIN = 1000;
float STRINMAX = 1990;
float STRINMIN = 1050;
#define THR_CH  1
#define STR_CH  0

// our servo # counter
uint8_t servonum = 0;

int thr_input;
int str_input;
int mode_input;

float thr_output = 0;
int thr_out = 0;
float str_output = 0;
int str_out;

#define THROTTLEMIN  330


void setup() {
  Serial.begin(9600);
  
  pwm.begin();
  
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates

  pinMode(2,INPUT);
  pinMode(4,INPUT);
  pinMode(6,INPUT);

  delay(10);
}

void loop() 
{
  thr_input = pulseIn(4,HIGH,1000000);
  str_input = pulseIn(2,HIGH,1000000);
  mode_input = pulseIn(6,HIGH,1000000);

  if (mode_input > 1500)
  {
    thr_output = ( (thr_input-THRINMIN)/(THRINMAX-THRINMIN) )*(THROTTLEMAX-THROTTLEMIN)+THROTTLEMIN;
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
